"""协议适配层 `protocol.adapt_model_response` + 内部解析助手测试。

覆盖：
- native 协议下原始 tool_calls 透传 + thought 来自 response.content
- native 协议空 tool_calls 时从 reasoning 提取草稿（含已知工具 / 未知工具）
"""

from __future__ import annotations

from agents.llm.protocol import (
    adapt_model_response,
    extract_reasoning_tool_call_draft,
)
from agents.llm.types import ModelResponse, ModelToolCall


def _resp(
    *,
    content: str = "",
    tool_calls: list[ModelToolCall] | None = None,
    reasoning: str = "",
    raw_response: str = "",
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tool_calls or [],
        raw_response=raw_response or content,
        raw_tool_calls=[],
        reasoning_content=reasoning,
    )


# ---------- native 协议 ----------


def test_native_protocol_passes_through_tool_calls() -> None:
    call = ModelToolCall(id="call_42", name="inspect_files", arguments={})
    response = _resp(content="thinking", tool_calls=[call])

    turn = adapt_model_response(
        response,
        turn_index=1,
        allowed_tool_names=("inspect_files",),
    )

    assert turn.parse_error is None
    assert turn.calls == (call,)
    assert turn.thought == "thinking"
    assert turn.reasoning_draft is None


def test_native_protocol_empty_calls_recovers_from_reasoning() -> None:
    reasoning = (
        "<tool_call>\n"
        "<function=read_doc>\n"
        "<parameter=path>knowledge.md</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )
    response = _resp(content="", reasoning=reasoning)

    turn = adapt_model_response(
        response,
        turn_index=1,
        allowed_tool_names=("read_doc", "answer"),
    )

    assert turn.parse_error is None
    assert len(turn.calls) == 1
    assert turn.calls[0].id == "recovered:1"
    assert turn.calls[0].name == "read_doc"
    assert turn.calls[0].arguments == {"path": "knowledge.md"}


def test_native_protocol_empty_calls_recovers_from_content() -> None:
    content = (
        "<tool_call><function=execute_python>"
        "<parameter=code>print(1)</parameter>"
        "</function></tool_call>"
    )
    response = _resp(content=content, reasoning="")

    turn = adapt_model_response(
        response,
        turn_index=5,
        allowed_tool_names=("execute_python", "answer"),
    )

    assert turn.parse_error is None
    assert len(turn.calls) == 1
    assert turn.calls[0].id == "recovered:5"
    assert turn.calls[0].name == "execute_python"
    assert turn.calls[0].arguments == {"code": "print(1)"}


def test_native_protocol_content_takes_priority_over_reasoning() -> None:
    content = (
        "<tool_call><function=inspect_files><parameter=path>.</parameter></function></tool_call>"
    )
    reasoning = (
        "<tool_call><function=execute_python>"
        "<parameter=code>print(1)</parameter>"
        "</function></tool_call>"
    )
    response = _resp(content=content, reasoning=reasoning)

    turn = adapt_model_response(
        response,
        turn_index=2,
        allowed_tool_names=("inspect_files", "execute_python", "answer"),
    )

    assert len(turn.calls) == 1
    assert turn.calls[0].name == "inspect_files"


def test_native_protocol_empty_calls_skips_unknown_tool_draft() -> None:
    reasoning = (
        "<tool_call><function=unknown_tool><parameter=x>1</parameter></function></tool_call>"
    )
    response = _resp(content="", reasoning=reasoning)

    turn = adapt_model_response(
        response,
        turn_index=1,
        allowed_tool_names=("inspect_files",),
    )

    assert turn.calls == ()
    assert turn.reasoning_draft is None


def test_native_protocol_empty_calls_no_reasoning() -> None:
    response = _resp(content="", reasoning="")

    turn = adapt_model_response(
        response,
        turn_index=1,
        allowed_tool_names=("answer",),
    )

    assert turn.calls == ()
    assert turn.reasoning_draft is None


def test_extract_reasoning_draft_parses_code_param_verbatim() -> None:
    reasoning = (
        "<tool_call><function=execute_python>"
        "<parameter=code>import os\nprint(os.listdir('.'))</parameter>"
        "</function></tool_call>"
    )
    draft = extract_reasoning_tool_call_draft(reasoning, ("execute_python",))
    assert draft is not None
    assert draft["arguments"]["code"] == "import os\nprint(os.listdir('.'))"
