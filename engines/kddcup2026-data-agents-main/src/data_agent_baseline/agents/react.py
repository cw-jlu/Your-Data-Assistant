from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.memory.error_patterns import error_signature_for
from data_agent_baseline.tools.registry import ToolRegistry


# v7.1 (N-2): when the agent hits the SAME error class on TWO consecutive
# steps, we surface a one-line "you've hit this twice — change approach"
# advisory in the next observation. Caps at one "circuit-breaker" cue per
# error class so we don't spam the prompt if the agent really is stuck.
_REPEAT_ERROR_THRESHOLD = 2


def _last_consecutive_error_signature(state: AgentRuntimeState) -> tuple[str, str, int] | None:
    """Walk the tail of ``state.steps`` and return ``(signature, action, count)``
    when the last ``_REPEAT_ERROR_THRESHOLD``+ steps share the same error class.

    Returns None when the streak is too short or interrupted by a successful
    step. We use this to inject a one-shot circuit-breaker into the next
    observation prompt.
    """
    if len(state.steps) < _REPEAT_ERROR_THRESHOLD:
        return None
    tail = list(state.steps)[-_REPEAT_ERROR_THRESHOLD:]
    if any(step.ok for step in tail):
        return None
    signatures: list[str] = []
    actions: list[str] = []
    for step in tail:
        observation = step.observation or {}
        if not isinstance(observation, dict):
            return None
        # Same extraction path the recorder uses — keeps signatures aligned.
        raw_error = ""
        if isinstance(observation.get("error"), str):
            raw_error = observation["error"]
        else:
            content = observation.get("content")
            if isinstance(content, dict):
                if isinstance(content.get("error"), str):
                    raw_error = content["error"]
                elif isinstance(content.get("stderr"), str) and content["stderr"]:
                    raw_error = content["stderr"]
        if not raw_error:
            return None
        signatures.append(error_signature_for(raw_error))
        actions.append(step.action or "")
    if not signatures:
        return None
    if len(set(signatures)) != 1:
        return None
    return signatures[0], actions[-1], len(tail)


def _repeat_error_advisory(signature: str, action: str, count: int) -> str:
    """Render the circuit-breaker advisory to inject into the next observation."""
    label = signature.replace("_", " ")
    return (
        f"REPEATED ERROR: the last {count} consecutive {action} calls hit"
        f" '{label}'. Do NOT retry the same approach. Inspect the data first"
        " (list_context / dataframe_describe / inspect_sqlite_schema) or switch"
        " tool. If you cannot recover, submit a best-effort answer rather than"
        " burning more steps."
    )


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    max_steps_by_difficulty: dict[str, int] | None = None
    # Number of corrective re-prompts after an initial parse failure.
    # 0 = current behaviour (one shot, then __error__ step). 2 = up to two
    # corrective re-prompts before giving up. Retries are NOT counted
    # toward the step budget.
    parse_retry_limit: int = 2


PARSE_RETRY_REMINDER = (
    "Your previous response could not be parsed. Re-emit the next step as a single "
    "JSON object with keys `thought`, `action`, and `action_input`, wrapped in exactly "
    "one ```json fenced block, with no prose before or after. "
    "`action_input` MUST be a JSON object (dict), never a bare string. "
    "For example: {\"code\": \"...\"} for execute_python, {\"path\": \"...\"} for read_csv/read_json/read_doc/dataframe_describe."
)


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


_PATH_TOOLS: frozenset[str] = frozenset({
    "read_csv",
    "read_json",
    "read_doc",
    "inspect_sqlite_schema",
    "dataframe_describe",
    "dataframe_head",
})


def _coerce_action_input(action: str, action_input: object) -> dict[str, object]:
    """Lenient wrap: many models emit a bare string for `action_input` instead of
    the expected JSON object. Recover the obvious common cases rather than
    burning a parse-retry budget item: most tools have a single dominant field.
    """
    if isinstance(action_input, dict):
        return action_input
    if isinstance(action_input, str):
        if action == "execute_python":
            return {"code": action_input}
        if action in _PATH_TOOLS:
            return {"path": action_input}
        if action == "list_context":
            # treat the bare string as a max_depth hint when numeric, else default
            try:
                return {"max_depth": int(action_input)}
            except ValueError:
                return {}
    raise ValueError("action_input must be a JSON object.")


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
    coerced_input = _coerce_action_input(action, action_input)

    return ModelStep(
        thought=thought,
        action=action,
        action_input=coerced_input,
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
        policy_hints: tuple[str, ...] | None = None,
        preferred_tools: tuple[str, ...] | None = None,
        avoid_tools: tuple[str, ...] | None = None,
        max_steps_multiplier: float = 1.0,
    ) -> None:
        self.model = model
        self.tools = tools
        self.config = config or ReActAgentConfig()
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT
        # v7 memory-layer policy injection. None / 1.0 = no-op (legacy v6 behavior).
        self._policy_hints = tuple(policy_hints or ())
        self._preferred_tools = tuple(preferred_tools or ())
        self._avoid_tools = tuple(avoid_tools or ())
        self._max_steps_multiplier = float(max_steps_multiplier or 1.0)

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        messages.append(
            ModelMessage(
                role="user",
                content=build_task_prompt(
                    task,
                    policy_hints=self._policy_hints,
                    preferred_tools=self._preferred_tools,
                    avoid_tools=self._avoid_tools,
                ),
            )
        )
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        # v7.1 N-2: if the tail of the trace shows a repeating error class,
        # inject a one-line circuit-breaker BEFORE the next model turn so the
        # agent stops flailing on the same broken approach.
        repeat = _last_consecutive_error_signature(state)
        if repeat is not None:
            signature, action, count = repeat
            messages.append(
                ModelMessage(
                    role="user",
                    content=_repeat_error_advisory(signature, action, count),
                )
            )
        return messages

    def _resolve_max_steps(self, task: PublicTask) -> int:
        overrides = self.config.max_steps_by_difficulty
        base: int
        if overrides:
            override = overrides.get(task.difficulty)
            if isinstance(override, int) and override > 0:
                base = override
            else:
                base = self.config.max_steps
        else:
            base = self.config.max_steps
        # v7 memory-layer scaling — heavy tasks get +25% step budget when
        # the matching ShapePolicy says so. Clamp at 1 to keep `for` loop safe.
        scaled = max(1, int(round(base * self._max_steps_multiplier)))
        return scaled

    def _complete_and_parse(
        self,
        base_messages: list[ModelMessage],
    ) -> tuple[ModelStep | None, list[str], list[str]]:
        """Call the model and parse. Retry with a corrective reminder on parse failure.

        Returns (model_step, raw_responses, parse_errors). parse_errors is empty
        on success. Retries are bounded by `config.parse_retry_limit` and do
        NOT consume entries in the agent's step counter.
        """
        retry_limit = max(0, int(self.config.parse_retry_limit))
        raw_responses: list[str] = []
        parse_errors: list[str] = []
        messages = list(base_messages)

        for attempt in range(retry_limit + 1):
            raw = self.model.complete(messages)
            raw_responses.append(raw)
            try:
                return parse_model_step(raw), raw_responses, parse_errors
            except Exception as exc:  # noqa: BLE001
                parse_errors.append(str(exc))
                if attempt >= retry_limit:
                    break
                messages = [
                    *messages,
                    ModelMessage(role="assistant", content=raw),
                    ModelMessage(role="user", content=PARSE_RETRY_REMINDER),
                ]
        return None, raw_responses, parse_errors

    def run(
        self,
        task: PublicTask,
        *,
        deadline: float | None = None,
    ) -> AgentRunResult:
        """Run the ReAct loop, optionally bounded by an absolute wall-clock deadline.

        ``deadline`` is a ``time.perf_counter()`` value. When set, the loop
        checks before each step and exits with ``failure_reason``
        ``"Agent exceeded per-sample deadline."`` if exceeded. This lets
        ``SelfConsistencyAgent`` budget k samples within one subprocess
        timeout instead of running k full ReAct passes serially and
        SIGKILL'ing in the middle of the second one.
        """
        state = AgentRuntimeState()
        max_steps = self._resolve_max_steps(task)
        try:
            for step_index in range(1, max_steps + 1):
                if deadline is not None and perf_counter() >= deadline:
                    state.failure_reason = "Agent exceeded per-sample deadline."
                    break
                base_messages = self._build_messages(task, state)
                model_step, raw_responses, parse_errors = self._complete_and_parse(base_messages)
                final_raw = raw_responses[-1] if raw_responses else ""

                if model_step is None:
                    observation: dict[str, object] = {
                        "ok": False,
                        "error": "Model output could not be parsed after retries.",
                        "parse_errors": parse_errors,
                        "parse_attempts": len(raw_responses),
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought="",
                            action="__error__",
                            action_input={},
                            raw_response=final_raw,
                            observation=observation,
                            ok=False,
                        )
                    )
                    continue

                try:
                    tool_result = self.tools.execute(task, model_step.action, model_step.action_input)
                    observation = {
                        "ok": tool_result.ok,
                        "tool": model_step.action,
                        "content": tool_result.content,
                    }
                    if len(raw_responses) > 1:
                        observation["parse_retries"] = len(raw_responses) - 1
                    step_record = StepRecord(
                        step_index=step_index,
                        thought=model_step.thought,
                        action=model_step.action,
                        action_input=model_step.action_input,
                        raw_response=final_raw,
                        observation=observation,
                        ok=tool_result.ok,
                    )
                    state.steps.append(step_record)
                    if tool_result.is_terminal:
                        state.answer = tool_result.answer
                        state.normalized_answer = tool_result.normalized_answer
                        break
                except Exception as exc:
                    observation = {
                        "ok": False,
                        "error": str(exc),
                    }
                    state.steps.append(
                        StepRecord(
                            step_index=step_index,
                            thought=model_step.thought,
                            action=model_step.action,
                            action_input=model_step.action_input,
                            raw_response=final_raw,
                            observation=observation,
                            ok=False,
                        )
                    )
        finally:
            cleanup = getattr(self.tools, "cleanup_task", None)
            if callable(cleanup):
                cleanup(task.task_id)

        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            normalized_answer=state.normalized_answer,
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
