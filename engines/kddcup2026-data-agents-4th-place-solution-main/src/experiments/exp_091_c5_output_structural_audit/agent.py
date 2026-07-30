from __future__ import annotations

import json
import re
from dataclasses import dataclass

from kobushi_core.model import ModelAdapter, ModelMessage, ModelStep
from kobushi_core.benchmark.schema import PublicTask

from experiments.exp_091_c5_output_structural_audit.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from experiments.exp_091_c5_output_structural_audit.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    StepRecord,
)
from experiments.exp_091_c5_output_structural_audit.tools.registry import ToolRegistry

# C5 post-answer structural audit: deterministic Python check, NO LLM call.
# Invariant: step-0 plan column_count=N => final answer must have <= N cols.
# - Equal or fewer cols: silent pass-through (revision-down is allowed; this
#   was exp_050's failure mode -- "COMMITMENT" wording blocked legitimate revisions).
# - Plan unparsable or cols <= planned: silent pass-through (no generic warning).
# - 1-retry budget (state.audit_retried); second submit always accepted.

_PLAN_COL_COUNT_RE = re.compile(r"column_count\s*[:=]\s*(\d+)", re.IGNORECASE)


def _extract_planned_column_count(steps: list[StepRecord]) -> int | None:
    """Parse step-0 thought for `column_count: N` (Rule 1 plan). None if absent."""
    if not steps:
        return None
    match = _PLAN_COL_COUNT_RE.search(steps[0].thought or "")
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _audit_terminal_answer(state: AgentRuntimeState, answer) -> str | None:
    """C5-c: return warning when len(answer.columns) > planned_N, else None (silent)."""
    if answer is None:
        return None
    planned = _extract_planned_column_count(state.steps)
    if planned is None:
        return None
    actual = len(answer.columns) if answer.columns is not None else 0
    if actual > planned:
        return (
            f"Audit-Warning: step-0 plan declared column_count={planned} but answer has "
            f"{actual} columns. Drop the {actual - planned} extra column(s) and re-submit."
        )
    return None


# Tools that produce a terminal answer — blocked during early exploration.
ANSWER_TOOLS: frozenset[str] = frozenset({"answer", "answer_from_python", "answer_from_sql"})


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    min_steps: int = 4  # Block answer tools for first N steps (0 = disabled).


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    # Case 3: unclosed fence — model outputs ```json {...} without closing ```
    unclosed_match = re.search(r"```(?:json)?\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if unclosed_match is not None:
        return unclosed_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r'[}\"\'\`\s]', "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return payload


def parse_model_step(raw_response: str) -> ModelStep:
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    # qwen3.5-a3b often passes Python code as a bare string instead of {"code": "..."}
    if isinstance(action_input, str):
        action_input = {"code": action_input}
    elif not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object or a plain string.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=action_input,
        raw_response=raw_response,
    )


class ReActAgent:
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        preamble: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.preamble = preamble

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        task_prompt = build_task_prompt(task)
        if self.preamble:
            task_prompt = f"{self.preamble}\n\n---\n\n{task_prompt}"
        messages.append(ModelMessage(role="user", content=task_prompt))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        for step_index in range(1, self.config.max_steps + 1):
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                model_step = parse_model_step(raw_response)
                # Early-answer guard: block answer tools for first min_steps steps.
                if step_index <= self.config.min_steps and model_step.action in ANSWER_TOOLS:
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=raw_response,
                            observation={
                                "ok": False,
                                "error": (
                                    f"Too early to answer (step {step_index} of "
                                    f"{self.config.min_steps} required exploration steps). "
                                    "Use execute_python or execute_sqlite to explore data first."
                                ),
                            },
                            ok=False,
                        )
                    )
                    continue  # resume outer for-loop; answer tool not executed
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)
                if tool_result.is_terminal:
                    audit_warning = _audit_terminal_answer(state, tool_result.answer)
                    if audit_warning and not state.audit_retried:
                        # Block finalization: inject warning, let agent revise next step.
                        state.audit_retried = True
                        state.steps.append(
                            StepRecord(
                                step_index=step_index,
                                thought=model_step.thought,
                                action=model_step.action,
                                action_input=model_step.action_input,
                                raw_response=raw_response,
                                observation={
                                    "ok": False,
                                    "tool": model_step.action,
                                    "audit_warning": audit_warning,
                                },
                                ok=False,
                            )
                        )
                        # Remove the preceding success record to avoid duplicate step_index.
                        if len(state.steps) >= 2 and state.steps[-2].step_index == step_index:
                            del state.steps[-2]
                        continue
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            preamble=self.preamble,
        )
