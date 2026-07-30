"""Task-level user prompt builders."""

from __future__ import annotations

from agents.benchmark.schema import PublicTask

_PREACT_PLAN_PROMPT = (
    "Before acting, outline a brief plan: "
    "(1) which files/tables contain the needed data, "
    "(2) what joins/filters/aggregations are needed, "
    "(3) what the final answer columns should be. "
    "Then start with explore."
)


def build_task_prompt(task: PublicTask, *, preact: bool = False) -> str:
    """任务起始的 user 消息——只放 question，其余全在 system prompt。"""
    text = f"Question: {task.question}"
    if preact:
        text = f"{text}\n\n{_PREACT_PLAN_PROMPT}"
    return text
