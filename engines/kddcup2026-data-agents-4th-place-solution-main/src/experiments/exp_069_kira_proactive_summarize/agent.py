from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass

from kobushi_core.model import ModelAdapter, ModelMessage, ModelStep
from kobushi_core.benchmark.schema import PublicTask

from experiments.exp_069_kira_proactive_summarize.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from experiments.exp_069_kira_proactive_summarize.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    StepRecord,
)
from experiments.exp_069_kira_proactive_summarize.tools.registry import ToolRegistry

# Chars-per-token rough estimate (no tiktoken dependency).
_CHARS_PER_TOKEN: float = 3.5
# Model context window (tokens). Qwen3.5-35b-A3B: 32k.
_CTX_LIMIT_TOKENS: int = 32_768
_CTX_LIMIT_CHARS: int = int(_CTX_LIMIT_TOKENS * _CHARS_PER_TOKEN)  # ~114_688

_SUMMARIZE_PROMPT = (
    "Summarize the key findings from the agent steps below in bullet form. Include:\n"
    "- Files/tables examined and their schemas (column names and types)\n"
    "- Key data values, counts, or statistical results discovered\n"
    "- SQL or Python queries that worked and their results\n"
    "- Dead ends or errors encountered, to avoid repeating them\n"
    "Keep it under 400 words. Steps to summarize:\n\n"
    "{steps_text}"
)


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    context_summarize_threshold: float = 0.80  # fraction of _CTX_LIMIT_CHARS


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

    def _estimate_context_chars(self, task: PublicTask, state: AgentRuntimeState) -> int:
        """Rough char count for the full message list _build_messages would produce."""
        messages = self._build_messages(task, state)
        return sum(len(m.content) for m in messages)

    def _compress_old_steps(
        self, task: PublicTask, state: AgentRuntimeState
    ) -> AgentRuntimeState:
        """Replace all-but-last-4 steps with a single LLM-generated summary step.

        On any failure, return state unchanged to avoid introducing missing_prediction.
        """
        steps = state.steps
        keep_recent = 4
        if len(steps) <= keep_recent:
            return state
        old_steps = steps[:-keep_recent]
        recent_steps = steps[-keep_recent:]

        lines: list[str] = []
        for s in old_steps:
            obs_text = (
                s.observation.get("content")
                or s.observation.get("error")
                or str(s.observation)
            )
            lines.append(f"Step {s.step_index} ({s.action}): {str(obs_text)[:800]}")
        steps_text = "\n".join(lines)

        try:
            summary_raw = self.model.complete(
                [ModelMessage(role="user", content=_SUMMARIZE_PROMPT.format(steps_text=steps_text))]
            )
        except Exception:
            return state  # summarize failed; leave untouched

        summary_step = StepRecord(
            step_index=old_steps[0].step_index,
            thought="[proactive_summarize] Previous observations compressed.",
            action="__summarize__",
            action_input={},
            raw_response=summary_raw,
            observation={"ok": True, "summary": summary_raw},
            ok=True,
        )
        return dataclasses.replace(state, steps=[summary_step] + list(recent_steps))

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
        threshold_chars = int(_CTX_LIMIT_CHARS * self.config.context_summarize_threshold)
        for step_index in range(1, self.config.max_steps + 1):
            # Proactive summarize: compress old steps when context grows too large.
            if (
                len(state.steps) > 4
                and self._estimate_context_chars(task, state) > threshold_chars
            ):
                state = self._compress_old_steps(task, state)
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                model_step = parse_model_step(raw_response)
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
