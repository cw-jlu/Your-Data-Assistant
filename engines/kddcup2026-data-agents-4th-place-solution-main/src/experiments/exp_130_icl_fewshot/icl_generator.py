"""ICL (In-Context Learning) SQL generator — DeepEye-SQL style.

Uses DAIL-SQL-retrieved 9-shot examples (= bird_dev_few_shots.json) for each
DABench task by mapping task_id → BIRD-dev question_id via project memory.

Pipeline (single LLM call, no ReAct loop):
  Question + schema_preview + 9 fewshot (question, sql) pairs → SQL → execute.

This is one of 3 attempts in adaptive_vote (= same n_attempts, mode diversity).
Other 2 attempts use the existing PhasedReActAgent for safety.

Source: github.com/HKUSTDial/DeepEye-SQL, arxiv 2510.17586
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.model import ModelAdapter, ModelMessage

from experiments.exp_130_icl_fewshot.tools.duckdb_unified import execute_sql


REPO = Path(__file__).resolve().parents[3]
_BIRD_MAPPING_FILE = REPO / "docs" / "_bird_mapping_data.json"
_FEWSHOT_FILE = REPO / "data" / "external" / "deepeye_fewshots" / "bird_dev_few_shots.json"


# =========================================================================
# Fewshot DB loading + lookup
# =========================================================================

@dataclass(frozen=True, slots=True)
class FewshotExample:
    question: str
    sql: str


def _load_fewshot_db() -> tuple[dict[str, int], dict[str, list[FewshotExample]]]:
    """Returns (task_id → bird_qid, bird_qid → fewshots)."""
    mapping_rows = json.loads(_BIRD_MAPPING_FILE.read_text())
    task_to_qid: dict[str, int] = {}
    for row in mapping_rows:
        qid = row.get("bird_qid")
        if qid is not None:
            task_to_qid[row["task_id"]] = int(qid)

    raw_few = json.loads(_FEWSHOT_FILE.read_text())
    qid_to_examples: dict[str, list[FewshotExample]] = {}
    for qid_str, examples in raw_few.items():
        qid_to_examples[qid_str] = [
            FewshotExample(question=str(ex["question"]), sql=str(ex["sql"]))
            for ex in examples
        ]
    return task_to_qid, qid_to_examples


# Module-level cache (= load once per process)
_TASK_TO_QID: dict[str, int] | None = None
_QID_TO_EXAMPLES: dict[str, list[FewshotExample]] | None = None


def get_fewshots(task_id: str, k: int = 9) -> list[FewshotExample]:
    """Returns up to k fewshot examples for the given DABench task_id.
    Empty list if mapping missing (= e.g., task_199 which has no BIRD qid)."""
    global _TASK_TO_QID, _QID_TO_EXAMPLES
    if _TASK_TO_QID is None or _QID_TO_EXAMPLES is None:
        _TASK_TO_QID, _QID_TO_EXAMPLES = _load_fewshot_db()
    qid = _TASK_TO_QID.get(task_id)
    if qid is None:
        return []
    examples = _QID_TO_EXAMPLES.get(str(qid), [])
    return examples[:k]


# =========================================================================
# Schema preview (= same approach as POC v3)
# =========================================================================

_VALUE_SAMPLE_LIMIT = 20  # per TEXT-like column
_TEXT_TYPES = ("VARCHAR", "TEXT", "CHAR", "STRING")


def _is_text_type(col_type: str) -> bool:
    return any(t in str(col_type).upper() for t in _TEXT_TYPES)


def _sample_distinct(context_dir: Path, table: str, column: str, limit: int) -> list[str]:
    """Top-N most frequent distinct values for a TEXT-like column.

    Ordered by frequency DESC so the most common (and hence most likely to be
    referenced) values appear first. Short symbols like 'p' / 'br' / '#' that
    general embedders handle poorly are still surfaced reliably this way."""
    try:
        # GROUP BY + COUNT(*) DESC = surface common values
        q = (
            f'SELECT "{column}" FROM "{table}" '
            f'WHERE "{column}" IS NOT NULL '
            f'GROUP BY "{column}" ORDER BY COUNT(*) DESC LIMIT {limit}'
        )
        res = execute_sql(context_dir, q, limit=limit)
        vals = []
        for row in res["rows"]:
            if not row:
                continue
            v = row[0]
            if v is None:
                continue
            s = str(v).strip()
            # Skip extremely long values (= prose-like) and empty
            if not s or len(s) > 80:
                continue
            vals.append(s)
        return vals
    except Exception:
        return []


def _build_schema_preview(context_dir: Path) -> str:
    """Schema overview enriched with sample values per TEXT-like column.

    Format:
      - <table>: <col1>:<type1> [sample: v1, v2, ...], <col2>:<type2>, ...

    The sample column values let the ICL prompt's LLM ground literals (e.g.
    'phosphorus' → 'p') rather than naively copying value literals from fewshot
    SQL examples (= which reference different databases).
    """
    try:
        res = execute_sql(context_dir, "SHOW TABLES")
        tables = [r[0] for r in res["rows"]]
    except Exception as exc:
        return f"[schema preview unavailable: {exc}]"
    lines = []
    for tbl in tables[:20]:
        try:
            cols = execute_sql(context_dir, f"DESCRIBE {tbl}")
            col_specs = []
            for r in cols["rows"][:30]:
                col_name, col_type = r[0], r[1]
                spec = f"{col_name}:{col_type}"
                # Inject samples only for TEXT-like columns
                if _is_text_type(col_type):
                    samples = _sample_distinct(context_dir, tbl, col_name, _VALUE_SAMPLE_LIMIT)
                    if samples:
                        # Compact format
                        sample_str = ", ".join(repr(s) for s in samples[:_VALUE_SAMPLE_LIMIT])
                        spec += f" [samples: {sample_str}]"
                col_specs.append(spec)
            lines.append(f"- {tbl}: {', '.join(col_specs)}")
        except Exception as exc:
            lines.append(f"- {tbl}: [error: {exc}]")
    return "\n".join(lines)


# =========================================================================
# ICL prompt + generation
# =========================================================================

_ICL_SYSTEM = """You are an expert SQL generator. You will be given:
1. A natural-language question
2. The schema of the available tables
3. A handful of example (question, SQL) pairs retrieved from a similar database

Your job: write the single SQL query that answers the user's question.

OUTPUT FORMAT:
- Exactly one fenced ```sql ... ``` block.
- The SQL must be a single SELECT/WITH statement (read-only).
- Use ONLY the table/column names from the provided schema. Do not invent names.
- Match the column shape the question asks for: usually one or two columns, not SELECT *.
- For "lowest/highest/min/max" questions, use `WHERE col = (SELECT MIN/MAX col FROM ...)` to keep ALL tied rows (= filter-back). Do not use `ORDER BY ... LIMIT 1`.
- For NULL-sensitive ORDER BY, add `WHERE col IS NOT NULL`.
- No `SELECT table.*` (= expand explicit columns).

Output ONLY the SQL fence. No explanation."""


def _format_fewshot(examples: list[FewshotExample]) -> str:
    """Format fewshot examples as compact (Q, SQL) blocks."""
    if not examples:
        return "(no fewshot examples available)"
    lines = []
    for i, ex in enumerate(examples, start=1):
        lines.append(f"### Example {i}\nQ: {ex.question}\nSQL: {ex.sql}\n")
    return "\n".join(lines)


def _extract_sql(raw: str) -> str:
    """Pull the first ```sql ... ``` block, or fall back to raw stripped."""
    m = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


@dataclass(frozen=True, slots=True)
class ICLResult:
    succeeded: bool
    sql: str = ""
    answer: AnswerTable | None = None
    failure_reason: str | None = None
    n_fewshots: int = 0


def generate_icl_attempt(task: PublicTask, model: ModelAdapter, k: int = 9) -> ICLResult:
    """Generate a single SQL via ICL fewshot prompting, then execute.

    Returns an ICLResult with the executed AnswerTable if successful.
    """
    examples = get_fewshots(task.task_id, k=k)
    schema = _build_schema_preview(task.context_dir)

    user_content = (
        f"# Question\n{task.question}\n\n"
        f"# Schema (= the available tables and columns)\n{schema}\n\n"
        f"# Fewshot examples (= similar (question, SQL) pairs)\n"
        f"{_format_fewshot(examples)}\n\n"
        f"Write the SQL for the question. Output only the ```sql ... ``` block."
    )
    raw = model.complete(
        [
            ModelMessage(role="system", content=_ICL_SYSTEM),
            ModelMessage(role="user", content=user_content),
        ],
        enable_thinking=False,
    )
    sql = _extract_sql(raw)
    if not sql:
        return ICLResult(succeeded=False, failure_reason="empty SQL from ICL", n_fewshots=len(examples))

    try:
        result = execute_sql(task.context_dir, sql, limit=10000)
    except Exception as exc:
        return ICLResult(succeeded=False, sql=sql, failure_reason=f"exec error: {exc}", n_fewshots=len(examples))

    columns = [str(c) for c in result.get("columns") or []]
    rows = [
        [("" if v is None else str(v)) for v in row]
        for row in (result.get("rows") or [])
    ]
    if not columns or not rows:
        return ICLResult(
            succeeded=False, sql=sql,
            failure_reason="empty result (no cols or no rows)",
            n_fewshots=len(examples),
        )
    answer = AnswerTable(columns=columns, rows=rows)
    return ICLResult(succeeded=True, sql=sql, answer=answer, n_fewshots=len(examples))
