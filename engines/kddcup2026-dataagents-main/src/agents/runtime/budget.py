"""Step-budget prompts and final-turn guardrails."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agents.llm.types import ModelMessage

_DEFAULT_TERMINAL_TOOL_NAMES = ("answer",)


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    remaining: int
    ratio: float


def compute_budget_status(*, effective_turns: int, max_steps: int) -> BudgetStatus:
    """Return the current budget ratio and remaining effective model turns."""
    ratio = effective_turns / max_steps if max_steps > 0 else 0
    return BudgetStatus(remaining=max_steps - effective_turns, ratio=ratio)


def _normalize_terminal_tool_names(
    terminal_tool_names: Sequence[str] | None,
) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(name for name in terminal_tool_names or () if name))
    return names or _DEFAULT_TERMINAL_TOOL_NAMES


def _format_terminal_tool_names(terminal_tool_names: Sequence[str]) -> str:
    quoted = [f"`{name}`" for name in terminal_tool_names]
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} or {quoted[-1]}"


def _terminal_call_phrase(terminal_tool_names: Sequence[str]) -> str:
    formatted = _format_terminal_tool_names(terminal_tool_names)
    if len(terminal_tool_names) == 1:
        return f"call {formatted}"
    return f"call one terminal tool ({formatted})"


def _terminal_artifact_name(terminal_tool_names: Sequence[str]) -> str:
    if terminal_tool_names == ("answer",):
        return "answer"
    if terminal_tool_names == ("report",):
        return "findings"
    return "result"


def _terminal_final_action_name(terminal_tool_names: Sequence[str]) -> str:
    if terminal_tool_names == ("report",):
        return "inspection"
    return "computation"


def build_budget_prompt(
    *,
    status: BudgetStatus,
    effective_turns: int,
    max_steps: int,
    warning_fired: bool,
    terminal_tool_names: Sequence[str] | None = None,
) -> tuple[ModelMessage | None, bool]:
    """Return an optional budget pressure message and the updated warning state."""
    terminal_names = _normalize_terminal_tool_names(terminal_tool_names)
    terminal_call = _terminal_call_phrase(terminal_names)
    artifact_name = _terminal_artifact_name(terminal_names)
    final_action_name = _terminal_final_action_name(terminal_names)
    failure_text = (
        "Failure to answer = score 0."
        if terminal_names == ("answer",)
        else "Failure to submit = score 0."
    )
    if status.ratio >= 0.90:
        return (
            ModelMessage(
                role="user",
                content=(
                    f"CRITICAL: {status.remaining} step(s) left. "
                    f"You MUST {terminal_call} NOW with whatever {artifact_name} you have. "
                    f"{failure_text}"
                ),
            ),
            warning_fired,
        )
    if not warning_fired and status.ratio >= 0.70:
        return (
            ModelMessage(
                role="user",
                content=(
                    f"BUDGET WARNING: {effective_turns}/{max_steps} steps used, "
                    f"{status.remaining} remaining. "
                    f"If you have prepared ANY {artifact_name}, {terminal_call} NOW — "
                    "an imperfect submission scores higher than no submission. "
                    f"If not, do ONE final {final_action_name} and submit."
                ),
            ),
            True,
        )
    return None, warning_fired


def is_final_step_blocking_active(*, status: BudgetStatus, last_step_retries: int) -> bool:
    """末步守卫是否处于激活区间（预算 ≥90%、剩余 ≤1、重试次数未耗尽）。

    不检查 calls 内容——仅判断"是否处于需要守卫干预的时间窗口"。
    供 agent.py 在末步混合轮场景下复用，无需重复阈值数字。
    """
    return status.remaining <= 1 and status.ratio >= 0.90 and last_step_retries < 2


def should_block_non_answer_final_turn(
    *,
    status: BudgetStatus,
    last_step_retries: int,
    calls: Sequence[Any],
    terminal_tool_names: Sequence[str] | None = None,
) -> bool:
    """Return whether the final-step guard should force an immediate answer call."""
    terminal_names = set(_normalize_terminal_tool_names(terminal_tool_names))
    return is_final_step_blocking_active(
        status=status, last_step_retries=last_step_retries
    ) and not any(getattr(call, "name", None) in terminal_names for call in calls)


def final_step_block_error(*, terminal_tool_names: Sequence[str] | None = None) -> str:
    """Return the observation error used when the final-step guard blocks a call."""
    terminal_names = _normalize_terminal_tool_names(terminal_tool_names)
    formatted = _format_terminal_tool_names(terminal_names)
    artifact_name = _terminal_artifact_name(terminal_names)
    if len(terminal_names) == 1:
        return (
            f"BLOCKED: final step must be the terminal tool call {formatted}. "
            f"Call {formatted} with your best {artifact_name} immediately."
        )
    return (
        f"BLOCKED: final step must be one terminal tool call ({formatted}). "
        "Call one with your best result immediately."
    )
