from __future__ import annotations

import json
import re
from dataclasses import dataclass

from kobushi_core.model import (
    ModelAdapter,
    ModelMessage,
    ModelStep,
    OpenAIModelAdapter,
)
from kobushi_core.benchmark.schema import PublicTask

from experiments.exp_038_function_calling.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from experiments.exp_038_function_calling.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    StepRecord,
)
from experiments.exp_038_function_calling.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16


def _strip_json_fence(raw_response: str) -> str:
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, object]:
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
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

    def _build_initial_messages(self, task: PublicTask) -> list[dict[str, object]]:
        """OpenAI-format initial messages (system + user). The conversation is then
        extended in-place by the run loop with assistant tool_calls and tool results.
        """
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        task_prompt = build_task_prompt(task)
        if self.preamble:
            task_prompt = f"{self.preamble}\n\n---\n\n{task_prompt}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": task_prompt},
        ]

    def run(self, task: PublicTask) -> AgentRunResult:
        state = AgentRuntimeState()
        if not isinstance(self.model, OpenAIModelAdapter):
            state.failure_reason = (
                "exp_038_function_calling requires OpenAIModelAdapter (native function-calling)."
            )
            return AgentRunResult(
                task_id=task.task_id,
                answer=state.answer,
                steps=list(state.steps),
                failure_reason=state.failure_reason,
                preamble=self.preamble,
            )

        messages = self._build_initial_messages(task)
        openai_tools = self.tools.to_openai_tools()

        for step_index in range(1, self.config.max_steps + 1):
            try:
                resp = self.model.complete_with_tools(messages, openai_tools)
            except Exception as exc:
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=str(exc),
                        observation={"ok": False, "error": str(exc)},
                        ok=False,
                    )
                )
                continue

            # If the model didn't call a tool, that's a "format-violation" step.
            # Re-prompt by appending its text + a hint, and loop again.
            if not resp.tool_calls:
                text = resp.text or ""
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought=text,
                        action="__no_tool_call__",
                        action_input={},
                        raw_response=text,
                        observation={
                            "ok": False,
                            "hint": (
                                "You must call one of the provided tools to make progress. "
                                "Do not respond in plain text — invoke a tool."
                            ),
                        },
                        ok=False,
                    )
                )
                # Append the assistant text + a user reminder.
                if resp.raw_assistant_message is not None:
                    messages.append(resp.raw_assistant_message)
                else:
                    messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Please call one of the available tools to continue. "
                            "Plain-text answers are not accepted."
                        ),
                    }
                )
                continue

            # Assistant message with tool_calls — append once, then process each call.
            if resp.raw_assistant_message is not None:
                messages.append(resp.raw_assistant_message)

            terminated = False
            for inv in resp.tool_calls:
                try:
                    tool_result = self.tools.execute(task, inv.name, inv.arguments)
                    observation = {
                        "ok": tool_result.ok,
                        "tool": inv.name,
                        "content": tool_result.content,
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=resp.text or "",
                            action=inv.name,
                            action_input=inv.arguments,
                            raw_response=str(resp.raw_assistant_message or ""),
                            observation=observation,
                            ok=tool_result.ok,
                        )
                    )
                    # Feed result back as a 'tool' message keyed to this tool_call_id.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": inv.id,
                            "content": json.dumps(observation, ensure_ascii=False, default=str),
                        }
                    )
                    if tool_result.is_terminal:
                        state.answer = tool_result.answer
                        terminated = True
                        break
                except Exception as exc:
                    err_obs = {"ok": False, "error": str(exc)}
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=resp.text or "",
                            action=inv.name,
                            action_input=inv.arguments,
                            raw_response=str(resp.raw_assistant_message or ""),
                            observation=err_obs,
                            ok=False,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": inv.id,
                            "content": json.dumps(err_obs, ensure_ascii=False, default=str),
                        }
                    )
            if terminated:
                break

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            preamble=self.preamble,
        )
