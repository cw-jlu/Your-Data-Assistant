"""ReAct 循环的步事件落定层。

把"哪个分支出错 / 哪个分支成功"从一组 `if action == "__error__"` 字面量沼泽换成
五个具名事件类型（`Literal` 闭集），driver 只 emit 事件，具体形状仍由本模块统一
序列化为 `StepRecord`，因此 wire format 保持字节稳定（D2）。

事件分类（与 `EventKind` 一一对应）：
- `model_error`:        `model.complete` 抛异常
- `parse_error`:        tool_call arguments JSON 解析失败
- `empty_tool_calls`:   native 协议返回空 tool_calls（含 reasoning 草稿兜底）
- `tool_error`:         tool handler 抛异常（含未知工具名 / 参数非法 / answer 校验失败）
- `observation`:        tool handler 返回（无论 ok 真假）

所有错误事件统一序列化为 `action="__error__"`；observation 事件的 `action` 是工具名。
`reasoning` / `usage` / `tool_call_id` / `raw_tool_calls` 仅由 driver 决定填什么
（同 turn 的并行 tool_calls 中只有"首条"承载 raw_tool_calls/usage/reasoning，
避免双计），本层只做透传。
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from agents.llm.protocol import (
    EMPTY_TOOL_CALL_ERROR,
    REASONING_DRAFT_RELAY_INSTRUCTION,
)
from agents.llm.types import TokenUsage
from agents.runtime.state import AgentRuntimeState, StepRecord

# 闭集事件名：driver 只能 emit 这五种，linter / pyright 会拒掉拼错
EventKind = Literal[
    "model_error",
    "parse_error",
    "empty_tool_calls",
    "tool_error",
    "observation",
]

# 与 agent.py 中保留同名签名：runner 注入的可选回调
StepCallback = Callable[[AgentRuntimeState], None]


@dataclass(frozen=True, slots=True)
class ModelErrorEvent:
    """`model.complete` 抛异常。

    `prompt_messages`：当步发给 LLM 的完整 messages（已序列化为 JSON 友好的 dict 列表）；
    API 失败路径下没有响应，故无 `response_message`。
    """

    error: str
    prompt_messages: list[dict[str, Any]] | None = None
    kind: EventKind = "model_error"


@dataclass(frozen=True, slots=True)
class ParseErrorEvent:
    """tool_call arguments JSON 解析失败。raw_response/usage/reasoning 来自原响应。

    `prompt_messages` / `response_message`：分别是当步发给 LLM 的 messages 与原始响应 dict；
    用于离线调试时回放当时的输入输出。
    """

    error: str
    raw_response: str
    reasoning: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    prompt_messages: list[dict[str, Any]] | None = None
    response_message: dict[str, Any] | None = None
    kind: EventKind = "parse_error"


@dataclass(frozen=True, slots=True)
class EmptyToolCallsEvent:
    """native 协议下模型返回空 tool_calls。

    `drafted_tool_call`：reasoning 中提取到的草稿（None 表示未捕到）。
    序列化时分两种 observation 文案：
    - 有草稿 → 短文案 + draft + relay instruction（让模型把草稿升格成真正的 tool_call）
    - 无草稿 → 长文案（更明确的纠偏引导）
    """

    thought: str
    raw_response: str
    reasoning: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    drafted_tool_call: dict[str, Any] | None = None
    prompt_messages: list[dict[str, Any]] | None = None
    response_message: dict[str, Any] | None = None
    kind: EventKind = "empty_tool_calls"


@dataclass(frozen=True, slots=True)
class ToolErrorEvent:
    """tool handler 抛异常（未知工具名 / 参数非法 / answer 校验失败等）。

    `tool_call_id` / `raw_tool_calls` / `reasoning` / `usage` 由 driver 按
    "lead-only"约定填充——同 turn 的并行 tool_calls 中只有首条带这些。
    `prompt_messages` / `response_message` 同样仅由 lead step 承载。
    """

    thought: str
    arguments: dict[str, Any]
    error: str
    raw_response: str
    tool_call_id: str | None = None
    raw_tool_calls: list[dict[str, Any]] | None = None
    reasoning: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    prompt_messages: list[dict[str, Any]] | None = None
    response_message: dict[str, Any] | None = None
    kind: EventKind = "tool_error"


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    """tool handler 成功返回（含 ok=False 但未抛异常的情况，如 execute_python 用户代码失败）。

    `tool_call_id` / `raw_tool_calls` / `reasoning` / `usage` 同 ToolErrorEvent 约定。
    `prompt_messages` / `response_message` 同样仅由 lead step 承载。
    """

    thought: str
    tool_name: str
    arguments: dict[str, Any]
    content: dict[str, Any]
    ok: bool
    raw_response: str
    tool_call_id: str | None = None
    raw_tool_calls: list[dict[str, Any]] | None = None
    reasoning: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    prompt_messages: list[dict[str, Any]] | None = None
    response_message: dict[str, Any] | None = None
    kind: EventKind = "observation"


StepEvent = (
    ModelErrorEvent | ParseErrorEvent | EmptyToolCallsEvent | ToolErrorEvent | ObservationEvent
)


def _empty_tool_calls_observation(drafted_tool_call: dict[str, Any] | None) -> dict[str, Any]:
    """构造 empty_tool_calls 事件的 observation 字典。

    - 有草稿：短文案 + draft + relay instruction
    - 无草稿：长文案（更明确的纠偏）
    """
    if drafted_tool_call is None:
        return {"ok": False, "error": EMPTY_TOOL_CALL_ERROR}
    return {
        "ok": False,
        "error": "Your previous turn did not produce an executable tool call.",
        "drafted_tool_call": drafted_tool_call,
        "instruction": REASONING_DRAFT_RELAY_INSTRUCTION,
    }


def _to_step_record(event: StepEvent, *, step_index: int, turn_index: int) -> StepRecord:
    """把事件序列化成 StepRecord（保持 wire format 字节稳定，D2）。"""
    if isinstance(event, ModelErrorEvent):
        # API 调用失败：没有响应可记，response_message 永远为 None；
        # prompt_messages 仍然有值，便于排查"哪条请求把模型炸了"
        return StepRecord(
            step_index=step_index,
            turn_index=turn_index,
            thought="",
            action="__error__",
            action_input={},
            raw_response="",
            observation={"ok": False, "error": event.error},
            ok=False,
            tool_call_id=None,
            raw_tool_calls=None,
            prompt_messages=event.prompt_messages,
            response_message=None,
        )
    if isinstance(event, ParseErrorEvent):
        return StepRecord(
            step_index=step_index,
            turn_index=turn_index,
            thought="",
            action="__error__",
            action_input={},
            raw_response=event.raw_response,
            observation={"ok": False, "error": event.error},
            ok=False,
            tool_call_id=None,
            raw_tool_calls=None,
            reasoning=event.reasoning,
            usage=event.usage,
            prompt_messages=event.prompt_messages,
            response_message=event.response_message,
        )
    if isinstance(event, EmptyToolCallsEvent):
        return StepRecord(
            step_index=step_index,
            turn_index=turn_index,
            thought=event.thought,
            action="__error__",
            action_input={},
            raw_response=event.raw_response,
            observation=_empty_tool_calls_observation(event.drafted_tool_call),
            ok=False,
            tool_call_id=None,
            raw_tool_calls=None,
            reasoning=event.reasoning,
            usage=event.usage,
            prompt_messages=event.prompt_messages,
            response_message=event.response_message,
        )
    if isinstance(event, ToolErrorEvent):
        return StepRecord(
            step_index=step_index,
            turn_index=turn_index,
            thought=event.thought,
            action="__error__",
            action_input=event.arguments,
            raw_response=event.raw_response,
            observation={"ok": False, "error": event.error},
            ok=False,
            tool_call_id=event.tool_call_id,
            raw_tool_calls=event.raw_tool_calls,
            reasoning=event.reasoning,
            usage=event.usage,
            prompt_messages=event.prompt_messages,
            response_message=event.response_message,
        )
    # ObservationEvent
    return StepRecord(
        step_index=step_index,
        turn_index=turn_index,
        thought=event.thought,
        action=event.tool_name,
        action_input=event.arguments,
        raw_response=event.raw_response,
        observation={"ok": event.ok, "tool": event.tool_name, "content": event.content},
        ok=event.ok,
        tool_call_id=event.tool_call_id,
        raw_tool_calls=event.raw_tool_calls,
        reasoning=event.reasoning,
        usage=event.usage,
        prompt_messages=event.prompt_messages,
        response_message=event.response_message,
    )


def record_step_event(
    state: AgentRuntimeState,
    event: StepEvent,
    *,
    step_index: int,
    turn_index: int,
    callback: StepCallback | None,
) -> StepRecord:
    """统一的"落定一步"通道：把事件序列化成 StepRecord，append 到 state，然后通知 callback。

    callback 异常被静默吞掉——观测回调挂掉不应该炸主任务。
    """
    record = _to_step_record(event, step_index=step_index, turn_index=turn_index)
    state.steps.append(record)
    if callback is not None:
        with contextlib.suppress(Exception):
            callback(state)
    return record
