"""Self-refinement loop (= ReFoRCE pillar (b)).

After the agent produces a terminal answer, classify it as suspicious if:
  - 0 rows returned (= filter likely too restrictive)
  - SQL execution failed (= already would not produce answer; reflected via
    `succeeded=False` on AgentRunResult)
  - Single-value answer for a question that asks for a list

If suspicious, re-invoke the agent with the prior failed answer + a hint
prompt, asking it to reconsider — typically by simplifying conditions or
re-checking the aggregate axis.

Refinement is bounded: at most `max_refines` retries (= default 1, since each
re-run is expensive).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kobushi_core.benchmark.schema import PublicTask
from kobushi_core.model import ModelAdapter

from experiments.exp_114_reforce_stack.agent import (
    ReActAgent,
    ReActAgentConfig,
)
from experiments.exp_114_reforce_stack.runtime import AgentRunResult


_LIST_QUESTION_PATTERNS = [
    r"\blist\b", r"\bwhich\b", r"\bwhat are\b", r"\bidentify\b",
    r"\bgive\b.*\b(all|the).*\b", r"\bnames? of\b", r"\bcountries?\b",
    r"\bnames\b", r"\btypes? of\b",
]


def is_list_question(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in _LIST_QUESTION_PATTERNS)


_SCALAR_QUESTION_PATTERNS = [
    r"\bhow many\b", r"\bcount\b", r"\btotal\b",
    r"\baverage\b", r"\bsum\b", r"\bmaximum\b", r"\bminimum\b",
    r"\bpercentage\b", r"\bratio\b",
]


def expects_scalar_answer(question: str) -> bool:
    q = question.lower()
    if any(re.search(p, q) for p in _SCALAR_QUESTION_PATTERNS):
        # but still might want grouped scalar — be conservative
        if re.search(r"\bfor each\b|\bper\b|\bgroup\b", q):
            return False
        return True
    return False


def is_suspicious(result: AgentRunResult, question: str) -> tuple[bool, str]:
    """Classify the answer as suspicious. Returns (is_suspicious, reason)."""
    if not result.succeeded or not result.answer:
        return True, "agent did not produce an answer (= run failure)"
    n_rows = len(result.answer.rows)
    n_cols = len(result.answer.columns)
    # Empty result is suspicious unless the question itself is a count of a
    # filter that may legitimately be 0 (= we can't tell, so flag for retry).
    if n_rows == 0:
        return True, "answer has 0 rows — filter likely too restrictive or aggregate-axis missing"
    # List-type questions returning a single row is often a misread.
    if n_rows == 1 and is_list_question(question) and not expects_scalar_answer(question):
        return True, f"question asks for a list but answer has only 1 row × {n_cols} cols"
    # Scalar question with multi-row answer = also suspicious
    if n_rows > 1 and expects_scalar_answer(question):
        return True, f"question expects a scalar but answer has {n_rows} rows"
    return False, ""


REFINEMENT_HINT_TEMPLATE = """\
Your prior answer attempt is below. It was flagged as SUSPICIOUS for the reason
shown. Reconsider the query carefully:

PRIOR ANSWER:
- columns: {columns}
- row count: {n_rows}
- first 3 rows: {first_rows}

REASON FLAGGED: {reason}

What to consider:
- If the result was EMPTY, your filter conditions may be too restrictive or
  the date/value format may not match the actual data. Try simplifying — start
  with a sanity-check query showing how many rows exist for each filter
  condition individually, then combine.
- If the result has the WRONG ROW COUNT shape (= list expected but got 1 row,
  or scalar expected but got many), re-read the question and re-check whether
  you should aggregate or not.
- If the result has the WRONG COLUMNS, re-read the question and ensure the
  SELECT names exactly the columns the question asks for.

Plan your refined query now and dispatch the FINAL answer with answer_from_sql.
"""


def build_refinement_preamble(
    base_preamble: str,
    prior_answer_columns: list[str],
    prior_answer_rows: list[list[Any]],
    flagged_reason: str,
) -> str:
    hint = REFINEMENT_HINT_TEMPLATE.format(
        columns=prior_answer_columns,
        n_rows=len(prior_answer_rows),
        first_rows=prior_answer_rows[:3],
        reason=flagged_reason,
    )
    return base_preamble + "\n\n---\n\n# REFINEMENT MODE\n" + hint


def run_with_refinement(
    *,
    task: PublicTask,
    model: ModelAdapter,
    tools,
    base_preamble: str,
    max_steps: int = 16,
    min_steps: int = 4,
    max_refines: int = 1,
) -> AgentRunResult:
    """Run the agent; on suspicious result, retry once with a refinement hint."""
    agent = ReActAgent(
        model=model,
        tools=tools,
        config=ReActAgentConfig(max_steps=max_steps, min_steps=min_steps),
        preamble=base_preamble,
    )
    result = agent.run(task)
    suspicious, reason = is_suspicious(result, task.question)
    refines = 0
    while suspicious and refines < max_refines:
        prior_cols = list(result.answer.columns) if result.answer else []
        prior_rows = list(result.answer.rows) if result.answer else []
        refined_preamble = build_refinement_preamble(
            base_preamble, prior_cols, prior_rows, reason
        )
        refined_agent = ReActAgent(
            model=model,
            tools=tools,
            config=ReActAgentConfig(max_steps=max_steps, min_steps=min_steps),
            preamble=refined_preamble,
        )
        refined_result = refined_agent.run(task)
        # Annotate metadata
        refined_result_steps = list(refined_result.steps)
        # If refined result is also empty/suspicious, keep the better of the two
        # (= prefer the one that actually succeeded with non-empty rows).
        new_suspicious, _ = is_suspicious(refined_result, task.question)
        if not new_suspicious or (
            refined_result.answer
            and len(refined_result.answer.rows) > len(prior_rows)
        ):
            result = refined_result
        suspicious = new_suspicious
        refines += 1
    return result
