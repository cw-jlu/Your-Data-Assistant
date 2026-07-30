"""事件落定层 `recorder.record_step_event` 测试。

每种事件 kind 走一遍：构造事件 → 经 recorder 落到 state.steps → 断言生成的
StepRecord 字段满足稳定序列化约定（错误事件统一 action="__error__"，
观察事件 action=tool 名）。

也覆盖 `step_callback` 的"每一步调用一次 + 异常被静默吞掉"。
"""

from __future__ import annotations

import pytest

from agents.llm.protocol import (
    EMPTY_TOOL_CALL_ERROR,
    REASONING_DRAFT_RELAY_INSTRUCTION,
)
from agents.llm.types import TokenUsage
from agents.runtime.recorder import (
    EmptyToolCallsEvent,
    ModelErrorEvent,
    ObservationEvent,
    ParseErrorEvent,
    ToolErrorEvent,
    record_step_event,
)
from agents.runtime.state import AgentRuntimeState


def _record_one(event, *, step_index: int = 1, turn_index: int = 1):
    """便捷：跑一次 record_step_event + 抓回唯一 step。"""
    state = AgentRuntimeState()
    record_step_event(
        state,
        event,
        step_index=step_index,
        turn_index=turn_index,
        callback=None,
    )
    assert len(state.steps) == 1
    return state.steps[0]


def test_model_error_event_serializes_to_error_action() -> None:
    step = _record_one(ModelErrorEvent(error="model.complete failed: ConnectionError"))

    assert step.action == "__error__"
    assert step.thought == ""
    assert step.action_input == {}
    assert step.raw_response == ""
    assert step.observation == {
        "ok": False,
        "error": "model.complete failed: ConnectionError",
    }
    assert step.ok is False
    assert step.tool_call_id is None
    assert step.raw_tool_calls is None
    # model_error 事件没有 reasoning / usage 字段
    assert step.reasoning == ""
    assert step.usage == TokenUsage()


def test_parse_error_event_carries_raw_response_and_usage() -> None:
    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    step = _record_one(
        ParseErrorEvent(
            error="action must be a non-empty string.",
            raw_response='```json\n{"action":""}\n```',
            reasoning="thinking...",
            usage=usage,
        )
    )

    assert step.action == "__error__"
    assert step.raw_response == '```json\n{"action":""}\n```'
    assert step.observation == {"ok": False, "error": "action must be a non-empty string."}
    assert step.reasoning == "thinking..."
    assert step.usage == usage


def test_empty_tool_calls_event_no_draft_uses_long_message() -> None:
    step = _record_one(
        EmptyToolCallsEvent(
            thought="I am thinking",
            raw_response="raw text",
            reasoning="",
            usage=TokenUsage(),
            drafted_tool_call=None,
        )
    )

    assert step.action == "__error__"
    assert step.thought == "I am thinking"
    assert step.observation == {"ok": False, "error": EMPTY_TOOL_CALL_ERROR}


def test_empty_tool_calls_event_with_draft_uses_short_message() -> None:
    draft = {
        "function": "inspect_files",
        "arguments": {},
        "source": "reasoning_content",
        "executed": False,
    }
    step = _record_one(
        EmptyToolCallsEvent(
            thought="",
            raw_response="",
            reasoning="<tool_call>...</tool_call>",
            usage=TokenUsage(),
            drafted_tool_call=draft,
        )
    )

    assert step.observation == {
        "ok": False,
        "error": "Your previous turn did not produce an executable tool call.",
        "drafted_tool_call": draft,
        "instruction": REASONING_DRAFT_RELAY_INSTRUCTION,
    }


def test_tool_error_event_preserves_call_metadata() -> None:
    raw_calls = [{"id": "c1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
    step = _record_one(
        ToolErrorEvent(
            thought="dispatch x",
            arguments={"foo": "bar"},
            error="Unknown tool: x",
            raw_response="model said x",
            tool_call_id="c1",
            raw_tool_calls=raw_calls,
            reasoning="reason",
            usage=TokenUsage(prompt_tokens=1),
        )
    )

    assert step.action == "__error__"
    assert step.action_input == {"foo": "bar"}
    assert step.observation == {"ok": False, "error": "Unknown tool: x"}
    assert step.tool_call_id == "c1"
    assert step.raw_tool_calls == raw_calls
    assert step.reasoning == "reason"


def test_observation_event_uses_tool_name_as_action() -> None:
    step = _record_one(
        ObservationEvent(
            thought="inspect it",
            tool_name="inspect_files",
            arguments={},
            content={"files": []},
            ok=True,
            raw_response="",
            tool_call_id="c1",
            raw_tool_calls=[{"id": "c1"}],
            reasoning="r",
            usage=TokenUsage(prompt_tokens=2),
        )
    )

    assert step.action == "inspect_files"
    assert step.action_input == {}
    assert step.observation == {
        "ok": True,
        "tool": "inspect_files",
        "content": {"files": []},
    }
    assert step.ok is True
    assert step.tool_call_id == "c1"


def test_observation_event_can_carry_ok_false_for_python_failure() -> None:
    step = _record_one(
        ObservationEvent(
            thought="run",
            tool_name="execute_python",
            arguments={"code": "raise"},
            content={"success": False, "stderr": "boom"},
            ok=False,
            raw_response="",
        )
    )

    assert step.action == "execute_python"
    assert step.ok is False
    assert step.observation["ok"] is False


def test_step_callback_invoked_each_record() -> None:
    state = AgentRuntimeState()
    callback_calls: list[int] = []

    def callback(s: AgentRuntimeState) -> None:
        callback_calls.append(len(s.steps))

    record_step_event(
        state,
        ModelErrorEvent(error="e1"),
        step_index=1,
        turn_index=1,
        callback=callback,
    )
    record_step_event(
        state,
        ModelErrorEvent(error="e2"),
        step_index=2,
        turn_index=2,
        callback=callback,
    )

    # 每次落定后立即触发；list 长度反映已 append 后的状态
    assert callback_calls == [1, 2]


def test_step_callback_exceptions_are_swallowed() -> None:
    state = AgentRuntimeState()

    def failing(_: AgentRuntimeState) -> None:
        raise RuntimeError("callback boom")

    # 不应该抛出
    record_step_event(
        state,
        ModelErrorEvent(error="x"),
        step_index=1,
        turn_index=1,
        callback=failing,
    )
    assert len(state.steps) == 1


def test_step_index_and_turn_index_pass_through() -> None:
    step = _record_one(ModelErrorEvent(error="e"), step_index=42, turn_index=7)
    assert step.step_index == 42
    assert step.turn_index == 7


@pytest.mark.parametrize(
    ("event", "expected_kind"),
    [
        (ModelErrorEvent(error="e"), "model_error"),
        (ParseErrorEvent(error="e", raw_response=""), "parse_error"),
        (
            EmptyToolCallsEvent(thought="", raw_response="", drafted_tool_call=None),
            "empty_tool_calls",
        ),
        (
            ToolErrorEvent(thought="", arguments={}, error="e", raw_response=""),
            "tool_error",
        ),
        (
            ObservationEvent(
                thought="",
                tool_name="x",
                arguments={},
                content={},
                ok=True,
                raw_response="",
            ),
            "observation",
        ),
    ],
)
def test_event_kind_literal_is_set_correctly(event, expected_kind: str) -> None:
    """每个事件类型自带正确的 `kind` literal——driver 不能把它们记错。"""
    assert event.kind == expected_kind
