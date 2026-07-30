"""ICL-enriched ReActAgent — single fewshot-aware attempt for mode diversity.

Replaces 1 of the 3 PhasedReActAgent attempts (= same total LLM load as v5).
The ICL agent has:
  - fewshot examples injected into system prompt
  - access to SQL execution tools for self-correction
  - shorter step budget (= 8) since fewshot pattern + 1-2 SQL iterations usually suffice

Key difference from single-shot ICL (= v1 of exp_130):
  - If first SQL errors or returns empty, agent can re-issue (= ReAct loop)
  - Tool set limited to SQL execution + answer submission (= no broad exploration)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.model import ModelAdapter

from experiments.exp_130_icl_fewshot.agent import (
    ReActAgent,
    ReActAgentConfig,
)
from experiments.exp_130_icl_fewshot.icl_generator import (
    _build_schema_preview,
    _format_fewshot,
    get_fewshots,
)
from experiments.exp_130_icl_fewshot.tools.registry import (
    ToolRegistry,
    create_default_tool_registry,
)


ICL_REACT_SYSTEM_PROMPT = """You are a SQL specialist. You will use ReAct (thought + action) to solve a question.

Your tools:
- describe_data(): inspect schema + sample values
- execute_sql(sql): run a SQL query (read-only) and see rows
- answer_from_sql(sql): submit final SQL — its rows become your final answer
- confirm_answer(): commit the most recent answer_from_sql as final

PROCESS:
1. Read the question + schema preview + 9 fewshot examples (= reference SQL patterns)
2. Try answer_from_sql with your best SQL
3. If error or unexpected result, refine and call answer_from_sql again (= max 2 refines)
4. Once satisfied, call confirm_answer

CRITICAL RULES:
- Use ONLY column/table names from the schema preview. Do NOT invent names.
- For "lowest/highest/min/max" with potential ties, use filter-back:
    WHERE col = (SELECT MIN/MAX col FROM t)   ← keeps ALL tied rows
  NOT `ORDER BY col LIMIT 1` (= drops ties).
- Result columns must match what the question asks for. Usually one column for "which X",
  not (X, metric_value) — do NOT add metric columns unless the question explicitly asks.
- For NULL-sensitive ORDER BY, add `WHERE col IS NOT NULL`.
- The fewshot examples reference DIFFERENT databases. Use them for SQL STRUCTURE PATTERNS
  only — do NOT copy table/column literal names from them.

Return format: one ```json fenced object with {thought, action, action_input} per step.
"""


def _build_icl_preamble(task: PublicTask, k: int = 9) -> str:
    """Preamble injected into the ICL ReAct agent.

    Includes (in order):
      1. Question
      2. Schema preview with sample values
      3. knowledge.md (= domain dictionary / value mappings — CRITICAL for tasks
         where the question uses domain terms whose meaning isn't obvious from
         schema alone, e.g. "severe thrombosis = Thrombosis=2")
      4. Fewshot reference examples (= structure patterns from similar questions)
    """
    examples = get_fewshots(task.task_id, k=k)
    schema = _build_schema_preview(task.context_dir)
    fewshot_block = _format_fewshot(examples)
    knowledge_path = task.context_dir / "knowledge.md"
    knowledge_block = ""
    if knowledge_path.exists():
        text = knowledge_path.read_text(errors="ignore")
        # Cap to ~3000 chars to keep prompt manageable (= phased agent uses
        # similar cap on its preamble's knowledge section).
        knowledge_block = (
            "# Domain knowledge guide (= read carefully, especially value mappings)\n"
            f"{text[:3000]}\n\n"
        )
    return (
        f"# Question\n{task.question}\n\n"
        f"# Schema (= the available tables and columns with sample values)\n{schema}\n\n"
        f"{knowledge_block}"
        f"# Fewshot reference examples (= structure patterns from similar questions)\n"
        f"{fewshot_block}\n\n"
        f"Now use the tools (describe_data, execute_sql, answer_from_sql, confirm_answer) "
        f"to compute the answer. Start by submitting your best SQL via answer_from_sql; refine if needed. "
        f"When a question uses domain terms (e.g. 'severe', 'cash withdrawal', 'normal range'), "
        f"check the domain knowledge guide ABOVE for the exact value mapping before writing SQL."
    )


@dataclass(frozen=True, slots=True)
class ICLReActResult:
    succeeded: bool
    answer: AnswerTable | None = None
    n_steps: int = 0
    failure_reason: str | None = None
    n_fewshots: int = 0


def run_icl_react_attempt(
    task: PublicTask,
    model: ModelAdapter,
    *,
    tools: ToolRegistry | None = None,
    k: int = 9,
    max_steps: int = 8,
) -> ICLReActResult:
    """Run the ICL ReAct agent on a task. Same return shape as a phased attempt."""
    if tools is None:
        tools = create_default_tool_registry(
            auditor_model=model,
            question_provider=lambda: task.question,
            context_dir=task.context_dir,
        )
    n_fewshots = len(get_fewshots(task.task_id, k=k))
    preamble = _build_icl_preamble(task, k=k)
    agent = ReActAgent(
        model=model,
        tools=tools,
        config=ReActAgentConfig(max_steps=max_steps, min_steps=1),
        system_prompt=ICL_REACT_SYSTEM_PROMPT,
        preamble=preamble,
    )
    result = agent.run(task)
    if not result.succeeded or not result.answer:
        return ICLReActResult(
            succeeded=False,
            n_steps=len(result.steps) if hasattr(result, 'steps') else 0,
            failure_reason=getattr(result, 'failure_reason', 'no answer'),
            n_fewshots=n_fewshots,
        )
    return ICLReActResult(
        succeeded=True,
        answer=result.answer,
        n_steps=len(result.steps) if hasattr(result, 'steps') else 0,
        n_fewshots=n_fewshots,
    )
