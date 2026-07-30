"""SQL voting at the terminal answer step (= ReFoRCE pillar (c) for inner SQL).

When the agent submits answer_from_sql, instead of executing one SQL, generate
K candidate SQLs at temperature 1.0 from the same prompt context, execute all,
and majority-vote on the result CSVs (= row-set match with numeric tolerance).
The most-voted result wins. Tie → fall back to the agent's own SQL.

Comparison logic mirrors ReFoRCE's `compare_pandas_table` in utils.py:
  - sort rows lexicographically (= ignore order)
  - tuple-wise comparison
  - numeric values matched within ±0.001
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_115_voted_sql.tools.duckdb_unified import (
    describe_catalog,
    execute_sql,
)


SQL_VOTER_SYSTEM = """\
You are a SQL specialist. The orchestrator agent has reached its final answer step.
Generate exactly ONE DuckDB SQL statement that produces the answer. Output a single
```sql fenced block with the query, no extra text.

Rules:
- Start with SELECT or WITH.
- Name every column explicitly. NEVER `SELECT *`.
- Match the answer's column_count to what the question asks for.
- For superlatives (lowest/highest/min/max), use filter-back not LIMIT 1:
      SELECT col FROM t WHERE val = (SELECT MIN(val) FROM t)
  This keeps tied rows.
- Apply ROUND() only at the outermost SELECT level.
- Cross-file JOIN is allowed (every CSV/JSON/sqlite is a DuckDB view).
- Return the answer for the question — DO NOT return debugging info.
"""


@dataclass(frozen=True, slots=True)
class SqlCandidate:
    sql: str
    columns: list[str] | None
    rows: list[list[Any]] | None
    error: str | None
    raw: str


def _extract_sql(text: str) -> str:
    m = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _normalize_value(v: Any) -> Any:
    """Normalize a single value for cross-candidate comparison."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            f = float(v)
            return round(f, 3)
        except Exception:
            return str(v).strip()
    s = str(v).strip()
    # Try numeric coercion
    try:
        return round(float(s), 3)
    except Exception:
        return s.lower()


def _result_signature(columns: list[str], rows: list[list[Any]]) -> tuple:
    """Stable, order-insensitive signature for a result table.

    Key idea: ignore column NAME order and ignore ROW order. Two results match
    if their multiset of (column-position-aware) row tuples is equivalent.
    """
    if not columns:
        return ("__empty__",)
    n_cols = len(columns)
    norm_rows: list[tuple] = []
    for r in rows:
        if len(r) != n_cols:
            continue
        norm_rows.append(tuple(_normalize_value(v) for v in r))
    norm_rows.sort()
    return (n_cols, len(norm_rows), tuple(norm_rows))


def _generate_one_candidate(
    *,
    task: PublicTask,
    sub_question: str,
    catalog_str: str,
    knowledge_slice: str,
    model: ModelAdapter,
    temperature: float,
    seed_idx: int,
) -> SqlCandidate:
    user_prompt = (
        f"## Question (orchestrator's stated final intent)\n{sub_question}\n\n"
        f"## DuckDB views available\n{catalog_str}\n\n"
        f"## knowledge.md (= caveats / definitions)\n{knowledge_slice}\n\n"
        "Output ONE ```sql block now."
    )
    # Adapter doesn't expose per-call temperature; we instead vary the prompt
    # nonce to encourage diversity even at the model's default temp. The
    # caller is expected to invoke this with a high-temperature ModelAdapter.
    messages = [
        ModelMessage(role="system", content=SQL_VOTER_SYSTEM),
        ModelMessage(role="user", content=user_prompt + f"\n\n<!--variant {seed_idx}-->"),
    ]
    try:
        raw = model.complete(messages)
    except Exception as exc:
        return SqlCandidate(sql="", columns=None, rows=None, error=f"LLM error: {exc}", raw="")
    sql = _extract_sql(raw)
    if not sql or not sql.strip().lower().startswith(("select", "with")):
        return SqlCandidate(sql=sql, columns=None, rows=None, error="not a SELECT/WITH", raw=raw)
    try:
        result = execute_sql(task.context_dir, sql, limit=10000)
    except Exception as exc:
        return SqlCandidate(sql=sql, columns=None, rows=None, error=f"exec error: {exc}", raw=raw)
    return SqlCandidate(
        sql=sql,
        columns=[str(c) for c in (result.get("columns") or [])],
        rows=[list(r) for r in (result.get("rows") or [])],
        error=None,
        raw=raw,
    )


def _knowledge_slice(task: PublicTask, max_chars: int = 4000) -> str:
    p = task.context_dir / "knowledge.md"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] + ("\n[...truncated]" if len(text) > max_chars else "")


@dataclass(frozen=True, slots=True)
class VotedResult:
    ok: bool
    columns: list[str]
    rows: list[list[Any]]
    winning_sql: str
    n_candidates: int
    n_successful: int
    cluster_sizes: list[int]   # sorted descending; e.g. [3, 1, 1] = 3-1-1 split
    has_unique_max: bool       # True iff exactly one cluster is the largest
    decision: str              # "unique_max" | "tie_llm_break" | "tie_fallback" | "all_failed"
    candidates: list[SqlCandidate]
    error: str | None = None


TIE_BREAK_SYSTEM = """\
You are picking the correct answer among candidate SQLs whose result rows tied
in a majority vote. Each candidate is shown with its SQL and its result rows.
Pick the candidate whose interpretation BEST matches the question.

Rules of judgement (in order):
1. Filter-back for superlatives (= preserves ties). Reject LIMIT 1 candidates.
2. Column count must match the question's plain reading.
3. Aggregate axis must match the question's grouping noun.
4. If still tied, prefer the candidate with fewer columns (= scoring favors
   fewer extras).

Output format: one line `WINNER: <index>` (= integer index of chosen candidate),
followed by one short sentence of reasoning.
"""


def _llm_tie_break(
    *, task: PublicTask, sub_question: str, tied: list[tuple[int, SqlCandidate]],
    model: ModelAdapter,
) -> int | None:
    """LLM tie-break. Returns the chosen candidate's index in `tied` (= local index),
    or None if the LLM produced no parseable choice."""
    blocks = []
    for local_i, (orig_i, c) in enumerate(tied):
        rows_preview = (c.rows or [])[:5]
        blocks.append(
            f"## Candidate {local_i}\n"
            f"SQL:\n```sql\n{c.sql}\n```\n"
            f"columns: {c.columns}\n"
            f"row_count: {len(c.rows or [])}\n"
            f"first 5 rows: {rows_preview}\n"
        )
    user = (
        f"## Question (orchestrator's stated intent)\n{sub_question}\n\n"
        + "\n".join(blocks)
        + f"\nReturn one line: WINNER: <0..{len(tied)-1}>"
    )
    raw = model.complete([
        ModelMessage(role="system", content=TIE_BREAK_SYSTEM),
        ModelMessage(role="user", content=user),
    ])
    m = re.search(r"WINNER\s*:\s*(\d+)", raw)
    if not m:
        return None
    idx = int(m.group(1))
    if 0 <= idx < len(tied):
        return idx
    return None


def voted_answer_from_sql(
    *,
    task: PublicTask,
    sub_question: str,
    model: ModelAdapter,
    k: int = 5,
    parallel: bool = True,
    tie_break_model: ModelAdapter | None = None,
    tie_fallback: str = "first_attempt",  # "first_attempt" | "abstain"
) -> VotedResult:
    """Generate K SQL candidates, execute all, vote on row-set.

    Decision tree (= ReFoRCE-style):
      - all candidates fail → ok=False (= "all_failed")
      - exactly one cluster has the largest size → accept ("unique_max")
      - multiple clusters tied at largest → LLM tie-break if tie_break_model
        provided ("tie_llm_break"); otherwise fall back per `tie_fallback`
        ("tie_fallback").
    """
    catalog_str = describe_catalog(task.context_dir)
    knowledge = _knowledge_slice(task)

    candidates: list[SqlCandidate] = []
    if parallel:
        with ThreadPoolExecutor(max_workers=k) as ex:
            futures = [
                ex.submit(
                    _generate_one_candidate,
                    task=task, sub_question=sub_question, catalog_str=catalog_str,
                    knowledge_slice=knowledge, model=model, temperature=1.0, seed_idx=i,
                )
                for i in range(k)
            ]
            for fut in as_completed(futures):
                candidates.append(fut.result())
    else:
        for i in range(k):
            candidates.append(_generate_one_candidate(
                task=task, sub_question=sub_question, catalog_str=catalog_str,
                knowledge_slice=knowledge, model=model, temperature=1.0, seed_idx=i,
            ))

    successful = [c for c in candidates if c.error is None and c.columns]
    if not successful:
        return VotedResult(
            ok=False, columns=[], rows=[], winning_sql="",
            n_candidates=k, n_successful=0, cluster_sizes=[],
            has_unique_max=False, decision="all_failed",
            candidates=candidates, error=f"all {k} candidates failed",
        )

    # Vote on result signatures
    sig_to_indices: dict[tuple, list[int]] = {}
    for idx, c in enumerate(successful):
        sig = _result_signature(c.columns, c.rows or [])
        sig_to_indices.setdefault(sig, []).append(idx)

    # Sort clusters by size descending
    clusters = sorted(sig_to_indices.items(), key=lambda kv: len(kv[1]), reverse=True)
    cluster_sizes = [len(idxs) for _, idxs in clusters]
    max_size = cluster_sizes[0]
    n_at_max = sum(1 for s in cluster_sizes if s == max_size)
    has_unique_max = n_at_max == 1

    if has_unique_max:
        winner_idx = clusters[0][1][0]
        winner = successful[winner_idx]
        return VotedResult(
            ok=True,
            columns=list(winner.columns or []),
            rows=list(winner.rows or []),
            winning_sql=winner.sql,
            n_candidates=k,
            n_successful=len(successful),
            cluster_sizes=cluster_sizes,
            has_unique_max=True,
            decision="unique_max",
            candidates=candidates,
        )

    # TIE — multiple clusters share the max size
    tied_clusters = [c for c in clusters if len(c[1]) == max_size]
    # Take one representative from each tied cluster for the tie-break
    tied_reps: list[tuple[int, SqlCandidate]] = []
    for _, idxs in tied_clusters:
        rep_idx = idxs[0]
        tied_reps.append((rep_idx, successful[rep_idx]))

    if tie_break_model is not None:
        chosen_local = _llm_tie_break(
            task=task, sub_question=sub_question, tied=tied_reps, model=tie_break_model,
        )
        if chosen_local is not None:
            winner_idx, winner = tied_reps[chosen_local]
            return VotedResult(
                ok=True,
                columns=list(winner.columns or []),
                rows=list(winner.rows or []),
                winning_sql=winner.sql,
                n_candidates=k,
                n_successful=len(successful),
                cluster_sizes=cluster_sizes,
                has_unique_max=False,
                decision="tie_llm_break",
                candidates=candidates,
            )

    # Tie fallback
    if tie_fallback == "abstain":
        return VotedResult(
            ok=False, columns=[], rows=[], winning_sql="",
            n_candidates=k, n_successful=len(successful),
            cluster_sizes=cluster_sizes, has_unique_max=False,
            decision="tie_fallback",
            candidates=candidates,
            error=f"vote tied: {n_at_max} clusters at size {max_size}",
        )
    # Default: first_attempt (= ReFoRCE's `random_vote_for_tie` behaviour, picks
    # the first-listed cluster which corresponds to a chronologically earlier
    # sample in our parallel batch).
    winner_idx, winner = tied_reps[0]
    return VotedResult(
        ok=True,
        columns=list(winner.columns or []),
        rows=list(winner.rows or []),
        winning_sql=winner.sql,
        n_candidates=k,
        n_successful=len(successful),
        cluster_sizes=cluster_sizes,
        has_unique_max=False,
        decision="tie_fallback",
        candidates=candidates,
    )
