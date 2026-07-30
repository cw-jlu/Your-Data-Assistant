"""Full-loop SQL sub-agent.

Unlike the 1-shot specialists in exp_112, this sub-agent runs its OWN
multi-step ReAct loop with restricted SQL-only tools. The orchestrator
sees only the sub-agent's final answer + a brief metadata summary —
the sub-agent's intermediate steps stay isolated.

Goal: protect the orchestrator's context from the schema dumps, error
messages, and intermediate query results that fill the agent loop.
"""
from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from typing import Any

from kobushi_core.benchmark.schema import PublicTask, TaskRecord
from kobushi_core.model import ModelAdapter

from experiments.exp_113_subagent_orchestrator.agent import (
    ReActAgent,
    ReActAgentConfig,
)
from experiments.exp_113_subagent_orchestrator.tools.duckdb_unified import describe_catalog


SUB_AGENT_SYSTEM_PROMPT = """\
You are a SQL specialist sub-agent. The ORCHESTRATOR has dispatched a focused
sub-question to you. Your single job: return the rows + columns that answer it.

Rules:
- All data lives in a unified DuckDB connection. Every CSV / JSON / sqlite-DB
  file in context is exposed as a view. Use `describe_data` to see them.
- Use `execute_sql` for inspection. Use `answer_from_sql` to terminate with the
  final query result — its output IS your return value to the orchestrator.
- Name every column explicitly in the final SELECT — no `SELECT *`. If the
  sub-question asks for ONE column, return ONE column.
- For superlative phrasing (lowest/highest/min/max), use filter-back not LIMIT 1
  to keep tied rows.
- Apply ROUND() only at the outermost SELECT.
- Read knowledge.md if the question references domain-specific terms (use `read_doc`).
- Keep the loop short (= max 8 steps). Be decisive.

Return format: always one ```json fenced JSON object with `thought`, `action`,
`action_input`. No surrounding text.
"""


SUB_AGENT_TOOL_NAMES = {
    "answer", "answer_from_sql", "describe_data", "execute_sql", "read_doc",
}


@dataclass(frozen=True, slots=True)
class SubAgentResult:
    ok: bool
    columns: list[str]
    rows: list[list[Any]]
    n_steps: int
    failure_reason: str | None = None


def _build_sub_task(task: PublicTask, sub_question: str) -> PublicTask:
    """Wrap the parent task with an overridden question for the sub-agent."""
    new_record = TaskRecord(
        task_id=f"{task.task_id}__sub",
        difficulty=task.record.difficulty,
        question=sub_question,
    )
    return PublicTask(record=new_record, assets=task.assets)


def _build_sub_preamble(task: PublicTask, sub_question: str) -> str:
    """Compact preamble: question + DuckDB catalog only (no full file dumps)."""
    catalog = describe_catalog(task.context_dir)
    return (
        f"# Sub-question (= focused task from orchestrator)\n"
        f"{sub_question}\n\n"
        f"# DuckDB views available\n{catalog}\n\n"
        f"# Reference\n"
        f"- For domain definitions / formulas, call `read_doc` with path `knowledge.md`.\n"
        f"- All sources are read-only. JOIN across views is allowed."
    )


def run_sql_subagent(
    *,
    task: PublicTask,
    sub_question: str,
    model: ModelAdapter,
    parent_tools=None,  # unused — kept for caller compatibility
    max_steps: int = 8,
) -> SubAgentResult:
    """Run a full-loop SQL sub-agent on a focused sub-question."""
    from experiments.exp_113_subagent_orchestrator.tools.registry import (
        create_subagent_tool_registry,
    )
    sub_tools = create_subagent_tool_registry()
    sub_task = _build_sub_task(task, sub_question)
    sub_preamble = _build_sub_preamble(task, sub_question)
    sub_agent = ReActAgent(
        model=model,
        tools=sub_tools,
        config=ReActAgentConfig(max_steps=max_steps, min_steps=2),
        system_prompt=SUB_AGENT_SYSTEM_PROMPT,
        preamble=sub_preamble,
    )
    result = sub_agent.run(sub_task)
    if not result.succeeded or not result.answer:
        return SubAgentResult(
            ok=False,
            columns=[],
            rows=[],
            n_steps=len(result.steps),
            failure_reason=result.failure_reason or "no answer produced",
        )
    return SubAgentResult(
        ok=True,
        columns=list(result.answer.columns),
        rows=[list(r) for r in result.answer.rows],
        n_steps=len(result.steps),
    )
