"""模型协议适配层（native → 统一 ProtocolTurn）。

把 `ModelResponse` 规整成同一个 `ProtocolTurn`，让 ReAct 循环驱动器
不再关心协议分支：
- native：原样使用 `response.tool_calls`；空时尝试从 reasoning 提取草稿，便于下一轮纠偏

`adapt_model_response` 是该层的唯一入口；其他符号都是内部解析助手或常量。
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from agents.llm.types import ModelResponse, ModelToolCall

_REASONING_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call\b[^>]*>(.*?)</tool_call>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REASONING_FUNCTION_PATTERN = re.compile(
    r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REASONING_PARAMETER_PATTERN = re.compile(
    r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>",
    flags=re.IGNORECASE | re.DOTALL,
)


# native 协议下，模型某轮没产出可执行 tool_call 时回灌给模型的纠偏文案。
# 长版本：模型甚至没有把调用塞进 reasoning（或拿不到草稿），需要更明确的引导。
EMPTY_TOOL_CALL_ERROR = (
    "Your previous turn did not produce an executable tool call. "
    "Any tool call written inside reasoning/thinking text was ignored, "
    "even if it looked syntactically complete. In this turn: do not "
    "draft <tool_call>, <function=...>, JSON actions, or executable "
    "code in reasoning. If you already decided which tool to call last "
    "turn, emit that call now as a structured tool call. If final answer "
    "is ready, call answer directly. You must call a tool."
)

# 短版本：reasoning 里捕到了草稿，附上草稿后只需提醒模型 "把它升格成真正的 tool_call"。
REASONING_DRAFT_RELAY_INSTRUCTION = (
    "A tool-call draft was found in reasoning_content, but reasoning drafts are not "
    "executable and were not run. On the next turn, emit the drafted function and "
    "arguments as a real structured tool_call through the native tool-calling "
    "interface. Do not put the call in reasoning/thinking text. If the final answer "
    "is ready, call answer directly."
)


@dataclass(frozen=True, slots=True)
class ProtocolTurn:
    """协议规整后的一轮模型输出。

    三种状态由 `calls` 判别：
    - `calls`：可派发的 tool_call 列表
    - `not calls`：模型回了空 tool_calls
        - `reasoning_draft is not None`：从 reasoning 抽到了草稿，可作为纠偏提示
        - `reasoning_draft is None`：完全空轮，需要长版纠偏文案

    `thought` 是要展示的"模型思考"字段，原样回填 `response.content`（可能是空串）。
    """

    calls: tuple[ModelToolCall, ...]
    thought: str
    parse_error: str | None = None
    reasoning_draft: dict[str, Any] | None = None


def _looks_like_json_value(value: str) -> bool:
    """Return whether a parameter value is worth trying to parse as JSON."""
    stripped = value.lstrip()
    if not stripped:
        return False
    return (
        stripped[0] in {'"', "{", "[", "-"}
        or stripped[0].isdigit()
        or stripped in {"true", "false", "null"}
    )


def _parse_reasoning_parameter_value(name: str, value: str) -> Any:
    """Parse one Qwen-style reasoning parameter conservatively."""
    stripped = value.strip()
    if name == "code":
        return stripped
    if _looks_like_json_value(stripped):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return stripped


def _parse_tool_call_tags(
    text: str,
    allowed_tool_names: Collection[str],
) -> tuple[str, dict[str, Any]] | None:
    """从文本中解析 Qwen 风格 <tool_call> 标签，返回 (function_name, arguments)。

    校验：函数名必须在 allowed_tool_names 中，参数必须全部由 <parameter> 标签覆盖
    （无残余文本）。解析失败返回 None。
    """
    if not text:
        return None

    tool_call_match = _REASONING_TOOL_CALL_PATTERN.search(text)
    if tool_call_match is None:
        return None

    tool_call_body = tool_call_match.group(1)
    function_match = _REASONING_FUNCTION_PATTERN.search(tool_call_body)
    if function_match is None:
        return None

    function_name = function_match.group(1).strip()
    if function_name not in allowed_tool_names:
        return None

    function_body = function_match.group(2)
    arguments: dict[str, Any] = {}
    for parameter_match in _REASONING_PARAMETER_PATTERN.finditer(function_body):
        parameter_name = parameter_match.group(1).strip()
        if not parameter_name:
            return None
        arguments[parameter_name] = _parse_reasoning_parameter_value(
            parameter_name,
            parameter_match.group(2),
        )

    unparsed_body = _REASONING_PARAMETER_PATTERN.sub("", function_body).strip()
    if unparsed_body:
        return None

    return function_name, arguments


def extract_reasoning_tool_call_draft(
    reasoning: str,
    allowed_tool_names: Collection[str],
) -> dict[str, Any] | None:
    """Extract the first complete Qwen-style tool-call draft from reasoning.

    This is deliberately best-effort and non-authoritative: it only helps the next
    model turn recover from a reasoning-only draft. The returned draft must never
    be executed by the runner.
    """
    parsed = _parse_tool_call_tags(reasoning, allowed_tool_names)
    if parsed is None:
        return None
    function_name, arguments = parsed
    return {
        "function": function_name,
        "arguments": arguments,
        "source": "reasoning_content",
        "executed": False,
    }


def adapt_model_response(
    response: ModelResponse,
    *,
    turn_index: int,
    allowed_tool_names: Collection[str],
) -> ProtocolTurn:
    """把 `ModelResponse` 规整成统一的 `ProtocolTurn`。

    直接使用 `response.tool_calls`；空时尝试从 reasoning 提取草稿。
    """
    calls = tuple(response.tool_calls)
    if not calls:
        for source in (response.content, response.reasoning_content):
            parsed = _parse_tool_call_tags(source, allowed_tool_names)
            if parsed is not None:
                fn_name, fn_args = parsed
                recovered = ModelToolCall(
                    id=f"recovered:{turn_index}",
                    name=fn_name,
                    arguments=fn_args,
                )
                return ProtocolTurn(calls=(recovered,), thought=response.content)
        draft = extract_reasoning_tool_call_draft(
            response.reasoning_content,
            allowed_tool_names,
        )
        return ProtocolTurn(calls=(), thought=response.content, reasoning_draft=draft)
    return ProtocolTurn(calls=calls, thought=response.content)
