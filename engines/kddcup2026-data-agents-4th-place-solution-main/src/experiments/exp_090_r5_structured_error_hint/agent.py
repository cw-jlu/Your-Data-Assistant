from __future__ import annotations

import json
import re
from dataclasses import dataclass

from kobushi_core.model import ModelAdapter, ModelMessage, ModelStep
from kobushi_core.benchmark.schema import PublicTask

from experiments.exp_090_r5_structured_error_hint.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from experiments.exp_090_r5_structured_error_hint.runtime import (
    AgentRunResult,
    AgentRuntimeState,
    StepRecord,
)
from experiments.exp_090_r5_structured_error_hint.tools.registry import ToolRegistry

# ============================================================================
# R5 structured execution feedback: exception-class -> concrete next-action hint
# ============================================================================
# Each hint is <= 180 chars and gives ONE specific recovery action.
# Unknown exception classes get NO hint (traceback only) -- exp_046 demonstrated
# that generic "Try: read the error..." hints (80% of cases) actively harm
# (lambda0.5 -0.047). Rule 11 already handles "no such table/column" so we
# skip those to avoid redundancy.
_HINT_BY_CLASS: dict[str, str] = {
    "KeyError": "KeyError usually means a missing DataFrame column or dict key. List available keys (df.columns.tolist() or dict.keys()) before retrying.",
    "AttributeError": "AttributeError: the object lacks that attribute. Print type(obj) and dir(obj) to find the correct method/attribute name.",
    "TypeError": "TypeError often means a non-numeric value reached arithmetic. Cast with pd.to_numeric(col, errors='coerce') or check dtypes via df.dtypes.",
    "ValueError": "ValueError means a value did not match expected format/range. Inspect a few sample values (df['col'].head() or repr(val)) before retrying the same parse/cast.",
    "ZeroDivisionError": "ZeroDivisionError. Filter out zero denominators (df = df[df['denom'] != 0]) or use np.where to guard the division.",
    "IndexError": "IndexError: index out of range. Check len()/shape before indexing; for empty results, verify the filter actually matches rows.",
    "ModuleNotFoundError": "Module unavailable. Pre-installed: pandas/numpy/sqlite3/json/csv/re/datetime/pathlib. SQLAlchemy is NOT installed -- use sqlite3 directly.",
    "ImportError": "ImportError: name not exported by that module. Check the module's actual API (use dir(module)) -- do not assume names from other libraries.",
    "SyntaxError": "Python SyntaxError. Do NOT retry the same code: re-read it for unmatched brackets/quotes/indentation, then rewrite minimally.",
    "FileNotFoundError": "File not found at that path. Call list_context first; all paths must be relative to the task context dir.",
}
# Note: 'OperationalError' (sqlite) deliberately omitted -- Rule 11 already
# directs the agent to inspect_sqlite_schema first.


def _hint_for_exception(exc: BaseException) -> str | None:
    """Return a concrete next-action hint for known exception classes only.
    Returns None for unknown classes -- by design, no generic fallback (exp_046 lesson).
    """
    return _HINT_BY_CLASS.get(type(exc).__name__)


def _hint_for_error_string(error_str: str) -> str | None:
    """Heuristically classify a tool's error string (when the tool returned
    ok=False with a string error rather than raising). Looks for the
    exception class name as it appears at the start of str(exc).
    Returns None when no class name is detected -- no generic fallback.
    """
    if not error_str:
        return None
    head = error_str.strip().split(":", 1)[0].strip()
    cls = head.rsplit(".", 1)[-1]
    return _HINT_BY_CLASS.get(cls)


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
                observation: dict[str, object] = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                # R5: when a tool returned ok=False with an error string in content,
                # attempt class-name detection and inject a hint. No generic fallback.
                if not tool_result.ok and isinstance(tool_result.content, dict):
                    err = tool_result.content.get("error")
                    if isinstance(err, str):
                        hint = _hint_for_error_string(err)
                        if hint:
                            observation["hint"] = hint
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
                observation: dict[str, object] = {
                    "ok": False,
                    "error": str(exc),
                }
                hint = _hint_for_exception(exc)
                if hint:
                    observation["hint"] = hint
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
