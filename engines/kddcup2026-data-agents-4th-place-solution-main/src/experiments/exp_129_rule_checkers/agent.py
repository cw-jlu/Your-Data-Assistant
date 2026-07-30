from __future__ import annotations

import json
import re
from dataclasses import dataclass

from kobushi_core.model import ModelAdapter, ModelMessage, ModelStep
from kobushi_core.benchmark.schema import PublicTask

from experiments.exp_129_rule_checkers.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from experiments.exp_129_rule_checkers.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    StepRecord,
)
from experiments.exp_129_rule_checkers.tools.registry import ToolRegistry

# Tools that produce a terminal answer — blocked during early exploration.
ANSWER_TOOLS: frozenset[str] = frozenset({"answer", "answer_from_sql"})


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
        collect_logprobs: bool = False,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        self.preamble = preamble
        # AIMO3-style entropy capture for entropy-weighted voting.
        # Requires the model adapter to expose complete_with_logprobs().
        self.collect_logprobs = collect_logprobs

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
        import os, sys
        trace_log = os.environ.get("KOBUSHI_TRACE_LOG")
        def _log(msg: str):
            if trace_log:
                with open(trace_log, "a") as f:
                    f.write(msg + "\n")

        state = AgentRuntimeState()
        # Decide once whether to use the logprobs path. Falls back to plain
        # complete() if the adapter does not implement complete_with_logprobs.
        use_logprobs = bool(self.collect_logprobs) and hasattr(
            self.model, "complete_with_logprobs"
        )
        for step_index in range(1, self.config.max_steps + 1):
            messages = self._build_messages(task, state)
            if use_logprobs:
                raw_response, top_buf = self.model.complete_with_logprobs(messages)
                if top_buf:
                    state.logprobs_buffer.extend(top_buf)
            else:
                raw_response = self.model.complete(messages)
            try:
                model_step = parse_model_step(raw_response)
                _log(
                    f"[{task.task_id}] step {step_index} action={model_step.action} "
                    f"thought={(model_step.thought or '')[:160].replace(chr(10),' | ')}"
                )
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
                                    "Use describe_data, execute_sql, or list_context to explore data first."
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
                _content_str = str(tool_result.content)[:300].replace("\n", " | ")
                _log(
                    f"[{task.task_id}] step {step_index} result ok={tool_result.ok} "
                    f"terminal={tool_result.is_terminal} content={_content_str}"
                )
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

        # Compute mean entropy over all collected output tokens (AIMO3-style).
        # H(token) = -Σ p_i log2 p_i; mean across N tokens.
        mean_h = float("inf")
        n_tok = len(state.logprobs_buffer)
        if n_tok:
            import math as _m
            total = 0.0
            for top in state.logprobs_buffer:
                h = 0.0
                for lp in top.values():
                    p = _m.exp(lp)
                    if p > 0:
                        h -= p * _m.log2(p)
                total += h
            mean_h = total / n_tok

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
            preamble=self.preamble,
            mean_entropy=mean_h,
            n_logprob_tokens=n_tok,
        )
