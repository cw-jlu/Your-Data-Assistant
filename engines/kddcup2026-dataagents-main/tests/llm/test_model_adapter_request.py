"""对 `OpenAIModelAdapter` / `NativeToolsOpenAIAdapter` 请求体与响应解析的单元测试。

真正的 OpenAI/vLLM 在 CI 上不可达，用 `monkeypatch` 把 `openai.OpenAI` 替换成
返回 `MagicMock` 的伪客户端，专注断言：
- 请求 kwargs 随配置正确组装（max_tokens / 采样字段 / extra_body 的条件下发）
- 响应侧把 `reasoning_content` 与 `usage`（含嵌套字段）解析到 `ModelResponse`

两个 adapter 共享 `_BaseOpenAIAdapter._build_request_kwargs`，请求组装断言统一
参数化在 adapter_cls 上跑两遍，防止任一 override 路径单独回归。

使用 `types.SimpleNamespace` 充当 SDK pydantic 对象的鸭子替身 —— 我们只读几个字段，
不需要真正的 pydantic 行为。
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from agents.llm import openai as model_mod
from agents.llm.openai import (
    BackendKind,
    NativeToolsOpenAIAdapter,
    OpenAIModelAdapter,
    ToolCallParseError,
    _extract_reasoning_content,
    _extract_usage,
    resolve_backend_kind,
)
from agents.llm.rate_limit import estimate_prompt_tokens
from agents.llm.types import ModelMessage, TokenUsage
from agents.tools.registry import (
    FunctionTool,
    ToolExecutionResult,
    ToolRegistry,
    create_default_tool_registry,
)


def _make_usage(
    *,
    prompt: int = 0,
    completion: int = 0,
    total: int = 0,
    cached: int | None = None,
    reasoning: int | None = None,
) -> types.SimpleNamespace:
    """构造伪 `response.usage`，可选填嵌套 details 对象。

    details 缺省为 None，用于测试"后端不返回嵌套字段时应归零"的分支。
    """
    usage_ns = types.SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )
    if cached is not None:
        usage_ns.prompt_tokens_details = types.SimpleNamespace(cached_tokens=cached)
    if reasoning is not None:
        usage_ns.completion_tokens_details = types.SimpleNamespace(reasoning_tokens=reasoning)
    return usage_ns


def _make_stream_chunks(
    *,
    content: str = "",
    reasoning_content: str | None = None,
    usage: Any = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> list[types.SimpleNamespace]:
    """构造伪流式 chunk 列表，模拟 `stream=True` 的逐块返回。"""
    chunks: list[types.SimpleNamespace] = []

    if reasoning_content:
        delta = types.SimpleNamespace(content=None, tool_calls=None)
        delta.reasoning_content = reasoning_content
        chunks.append(
            types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)], usage=None)
        )

    if content:
        delta = types.SimpleNamespace(content=content, tool_calls=None)
        chunks.append(
            types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)], usage=None)
        )

    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            tc_delta = types.SimpleNamespace(
                index=tc.get("_index", 0),
                id=tc.get("id"),
                type=tc.get("type"),
                function=types.SimpleNamespace(
                    name=fn.get("name"),
                    arguments=fn.get("arguments"),
                ),
            )
            delta = types.SimpleNamespace(content=None, tool_calls=[tc_delta])
            chunks.append(
                types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)], usage=None)
            )

    if not chunks:
        delta = types.SimpleNamespace(content="", tool_calls=None)
        chunks.append(
            types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)], usage=None)
        )

    final_chunk = types.SimpleNamespace(choices=[], usage=usage)
    chunks.append(final_chunk)

    return chunks


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, chunks: Any) -> MagicMock:
    """把 `model.OpenAI` 换成返回 MagicMock 的伪类，拦截 `chat.completions.create`。

    `chunks` 应为 `_make_stream_chunks(...)` 返回的列表。create 返回其迭代器，
    模拟流式响应。测试用 `create_mock.call_args.kwargs` 断言请求 kwargs。
    """
    create_mock = MagicMock(side_effect=lambda **_kw: iter(chunks))
    client_ns = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create_mock),
        )
    )
    monkeypatch.setattr(model_mod, "OpenAI", lambda **_: client_ns)
    return create_mock


ADAPTER_CLASSES = [
    pytest.param(OpenAIModelAdapter, id="openai"),
    pytest.param(NativeToolsOpenAIAdapter, id="native"),
]


def _adapter(adapter_cls: type, **overrides: Any) -> Any:
    """构造任一 adapter；native 额外注入默认工具注册表。"""
    defaults: dict[str, Any] = {
        "model": "m",
        "api_base": "http://x",
        "api_key": "k",
        "temperature": 0.0,
        "max_tokens": 0,
        "enable_thinking": None,
    }
    if adapter_cls is NativeToolsOpenAIAdapter:
        defaults["tools"] = create_default_tool_registry()
    defaults.update(overrides)
    return adapter_cls(**defaults)


# ---------- 基础 helper 的小单测：先保证 _extract_* 行为可靠 ----------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        pytest.param(
            types.SimpleNamespace(reasoning_content="pondering"), "pondering", id="direct-attr"
        ),
        # 没声明 reasoning_content，但底层 JSON 挂在 model_extra 上
        pytest.param(
            types.SimpleNamespace(model_extra={"reasoning_content": "hidden"}),
            "hidden",
            id="model-extra-fallback",
        ),
        pytest.param(
            types.SimpleNamespace(reasoning="direct-hidden"),
            "direct-hidden",
            id="direct-reasoning-alias",
        ),
        pytest.param(
            types.SimpleNamespace(model_extra={"reasoning": "hidden-alias"}),
            "hidden-alias",
            id="model-extra-reasoning-alias",
        ),
        pytest.param(types.SimpleNamespace(), "", id="absent"),
    ],
)
def test_extract_reasoning_content(message: Any, expected: str) -> None:
    assert _extract_reasoning_content(message) == expected


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        pytest.param(
            _make_usage(prompt=1200, completion=450, total=1650, cached=800, reasoning=300),
            TokenUsage(1200, 450, 1650, 800, 300),
            id="full-nested-payload",
        ),
        # 没有 details 嵌套对象 → cached / reasoning 归零，不抛错
        pytest.param(
            _make_usage(prompt=100, completion=50, total=150),
            TokenUsage(100, 50, 150, 0, 0),
            id="missing-details",
        ),
        pytest.param(None, TokenUsage(), id="missing-usage"),
        # DeepSeek V4 返回 prompt_cache_hit_tokens 在 usage 顶层
        pytest.param(
            types.SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                prompt_cache_hit_tokens=600,
                prompt_cache_miss_tokens=400,
                completion_tokens_details=types.SimpleNamespace(reasoning_tokens=50),
            ),
            TokenUsage(1000, 200, 1200, 600, 50),
            id="deepseek-top-level-cache-hit",
        ),
        # 同时存在嵌套 cached_tokens 和顶层 prompt_cache_hit_tokens 时取 max
        pytest.param(
            types.SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200,
                prompt_tokens_details=types.SimpleNamespace(cached_tokens=300),
                prompt_cache_hit_tokens=600,
            ),
            TokenUsage(1000, 200, 1200, 600, 0),
            id="prefers-larger-cached-value",
        ),
    ],
)
def test_extract_usage_payload_shapes(usage: Any, expected: TokenUsage) -> None:
    assert _extract_usage(types.SimpleNamespace(usage=usage)) == expected


# ---------- 请求 kwargs 组装：两个 adapter 共享 _build_request_kwargs ----------


@pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES)
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({"max_tokens": 4096}, {"max_tokens": 4096}, id="max-tokens"),
        pytest.param({"top_p": 0.95}, {"top_p": 0.95}, id="top-p"),
        pytest.param({"seed": 42}, {"seed": 42}, id="seed"),
        pytest.param({"presence_penalty": 1.5}, {"presence_penalty": 1.5}, id="presence-penalty"),
    ],
)
def test_adapter_sends_sampling_fields_when_set(
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls: type,
    overrides: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = _adapter(adapter_cls, **overrides)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    for key, value in expected.items():
        assert kwargs[key] == value


@pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES)
@pytest.mark.parametrize(
    ("overrides", "absent_key"),
    [
        # 0 = 不下发 max_tokens
        pytest.param({"max_tokens": 0}, "max_tokens", id="max-tokens-zero"),
        pytest.param({"temperature": None}, "temperature", id="temperature-none"),
        pytest.param({}, "top_p", id="top-p-unset"),
        pytest.param({"enable_thinking": None}, "extra_body", id="thinking-none"),
    ],
)
def test_adapter_omits_optional_fields_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls: type,
    overrides: dict[str, Any],
    absent_key: str,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = _adapter(adapter_cls, **overrides)
    adapter.complete([ModelMessage(role="user", content="hello")])

    assert absent_key not in create_mock.call_args.kwargs


@pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES)
@pytest.mark.parametrize(
    ("overrides", "expected_extra_body"),
    [
        # api_base="http://x" 不含 dashscope hint → 自动推断 vllm → 嵌套形式
        pytest.param(
            {"enable_thinking": True},
            {"chat_template_kwargs": {"enable_thinking": True}},
            id="thinking-true",
        ),
        # False 也要显式下发 —— 避免服务端默认 on 时用户的关闭意图被吞掉
        pytest.param(
            {"enable_thinking": False},
            {"chat_template_kwargs": {"enable_thinking": False}},
            id="thinking-false",
        ),
        # 关键回归：top_k / min_p 必须与 enable_thinking 在 extra_body 里共存而非覆盖
        pytest.param(
            {"enable_thinking": True, "top_k": 20, "min_p": 0.0},
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "top_k": 20,
                "min_p": 0.0,
            },
            id="thinking-merges-top-k",
        ),
        pytest.param(
            {"enable_thinking": None, "top_k": 20},
            {"top_k": 20},
            id="only-top-k-when-thinking-none",
        ),
    ],
)
def test_adapter_extra_body_composition(
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls: type,
    overrides: dict[str, Any],
    expected_extra_body: dict[str, Any],
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = _adapter(adapter_cls, **overrides)
    adapter.complete([ModelMessage(role="user", content="hello")])

    assert create_mock.call_args.kwargs["extra_body"] == expected_extra_body


def test_native_adapter_sends_tool_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """native function calling 的请求必须带 tools / tool_choice / parallel_tool_calls。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _adapter(NativeToolsOpenAIAdapter, max_tokens=8192)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["max_tokens"] == 8192
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True
    assert isinstance(kwargs["tools"], list) and len(kwargs["tools"]) > 0


def test_native_adapter_forced_tool_choice_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    forced = {"type": "function", "function": {"name": "report"}}
    adapter = _adapter(NativeToolsOpenAIAdapter, tool_choice=forced)
    adapter.complete([ModelMessage(role="user", content="hello")])

    assert create_mock.call_args.kwargs["tool_choice"] == forced


def _tiny_registry() -> ToolRegistry:
    """单工具 registry，schema 远小于默认 registry，用作 per-call override。"""
    return ToolRegistry(
        {
            "ping": FunctionTool(
                name="ping",
                description="tiny override tool",
                json_schema={"type": "object", "properties": {}},
                handler=lambda _t, _a: ToolExecutionResult(ok=True, content={}),
            )
        }
    )


def test_native_adapter_per_call_tools_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """per-call override registry：请求 kwargs 的 tools 等于 override 的渲染结果。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _adapter(NativeToolsOpenAIAdapter)
    override = _tiny_registry()
    adapter.complete([ModelMessage(role="user", content="hello")], tools=override)

    assert create_mock.call_args.kwargs["tools"] == override.to_openai_tools()


def test_native_adapter_same_registry_identity_hits_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tools 传构造期同一 registry 对象时走缓存路径（字节稳定的同一 payload 对象）。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    registry = create_default_tool_registry()
    adapter = _adapter(NativeToolsOpenAIAdapter, tools=registry)
    adapter.complete([ModelMessage(role="user", content="hello")], tools=registry)

    assert create_mock.call_args.kwargs["tools"] is adapter._openai_tools


def test_native_adapter_prompt_estimate_reflects_tools_override() -> None:
    messages = [ModelMessage(role="user", content="hello")]
    adapter = _adapter(NativeToolsOpenAIAdapter)

    default_estimate = adapter.estimate_prompt_tokens(messages)
    override_estimate = adapter.estimate_prompt_tokens(messages, tools=_tiny_registry())

    # override schema 远小于默认 registry，但仍大于纯 messages 估算
    assert override_estimate < default_estimate
    assert override_estimate > estimate_prompt_tokens(messages)


# ---------- 响应解析 ----------


@pytest.mark.parametrize("adapter_cls", ADAPTER_CLASSES)
def test_adapter_captures_reasoning_and_usage(
    monkeypatch: pytest.MonkeyPatch, adapter_cls: type
) -> None:
    chunks = _make_stream_chunks(
        content="final answer",
        reasoning_content="step-by-step thought",
        usage=_make_usage(prompt=100, completion=50, total=150, cached=40, reasoning=20),
    )
    _install_fake_openai(monkeypatch, chunks)
    adapter = _adapter(adapter_cls)

    result = adapter.complete([ModelMessage(role="user", content="hello")])

    assert result.content == "final answer"
    assert result.reasoning_content == "step-by-step thought"
    assert result.usage == TokenUsage(100, 50, 150, 40, 20)
    assert result.latency_ms >= 0


def test_native_adapter_malformed_tool_arguments_preserve_partial_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_tool_call = {
        "id": "bad_answer",
        "type": "function",
        "function": {
            "name": "answer",
            "arguments": '{"columns": ["x"], "rows": [["unterminated"]',
        },
    }
    chunks = _make_stream_chunks(
        content="submitting answer",
        reasoning_content="answer json got truncated",
        usage=_make_usage(prompt=10, completion=5, total=15, cached=3, reasoning=2),
        tool_calls=[{**raw_tool_call, "_index": 0}],
    )
    _install_fake_openai(monkeypatch, chunks)
    adapter = _adapter(NativeToolsOpenAIAdapter)

    with pytest.raises(ToolCallParseError) as exc_info:
        adapter.complete([ModelMessage(role="user", content="hello")])

    partial = exc_info.value.partial_response
    assert "Failed to parse tool_call arguments for 'answer'" in str(exc_info.value)
    assert partial.content == "submitting answer"
    assert partial.raw_response == "submitting answer"
    assert partial.raw_tool_calls == [raw_tool_call]
    assert partial.reasoning_content == "answer json got truncated"
    assert partial.usage == TokenUsage(10, 5, 15, 3, 2)
    assert partial.tool_calls == []
    assert partial.latency_ms >= 0


def test_native_adapter_prompt_estimate_includes_tool_schema() -> None:
    messages = [ModelMessage(role="user", content="hello")]
    adapter = _adapter(NativeToolsOpenAIAdapter)

    message_only = estimate_prompt_tokens(messages)
    estimated = adapter.estimate_prompt_tokens(messages)

    assert estimated > message_only
    assert estimated - message_only > 1_000


# ---------- backend_kind 矩阵：vllm vs dashscope，自动推断 vs 显式覆盖 ----------


@pytest.mark.parametrize(
    ("adapter_cls", "api_base", "backend_kind", "enable_thinking", "expected_extra_body"),
    [
        # api_base 含 aliyuncs.com → 自动推断 dashscope → enable_thinking 走顶层
        pytest.param(
            OpenAIModelAdapter,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            None,
            True,
            {"enable_thinking": True},
            id="openai-dashscope-auto-infer",
        ),
        # 显式 backend_kind="dashscope" 覆盖 localhost api_base 的 vllm 推断
        pytest.param(
            OpenAIModelAdapter,
            "http://localhost:8000/v1",
            "dashscope",
            True,
            {"enable_thinking": True},
            id="openai-explicit-dashscope-wins",
        ),
        # 显式 backend_kind="vllm" 覆盖 dashscope api_base 的自动推断 → 仍走嵌套
        pytest.param(
            OpenAIModelAdapter,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "vllm",
            True,
            {"chat_template_kwargs": {"enable_thinking": True}},
            id="openai-explicit-vllm-wins",
        ),
        # 两个 adapter 共用 _thinking_extra_body，显式 backend_kind 行为一致
        pytest.param(
            NativeToolsOpenAIAdapter,
            "http://x",
            "vllm",
            True,
            {"chat_template_kwargs": {"enable_thinking": True}},
            id="native-explicit-vllm",
        ),
        pytest.param(
            NativeToolsOpenAIAdapter,
            "http://x",
            "dashscope",
            False,
            {"enable_thinking": False},
            id="native-explicit-dashscope",
        ),
        # enable_thinking=None 时无论 backend_kind 是什么都不下发 extra_body，
        # 避免 backend_kind 单独下发误触发未定义 chat template 行为
        pytest.param(
            OpenAIModelAdapter,
            "http://x",
            "vllm",
            None,
            None,
            id="thinking-none-no-extra-body",
        ),
    ],
)
def test_backend_kind_thinking_placement_matrix(
    monkeypatch: pytest.MonkeyPatch,
    adapter_cls: type,
    api_base: str,
    backend_kind: str | None,
    enable_thinking: bool | None,
    expected_extra_body: dict[str, Any] | None,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = _adapter(
        adapter_cls,
        api_base=api_base,
        backend_kind=backend_kind,
        enable_thinking=enable_thinking,
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    if expected_extra_body is None:
        assert "extra_body" not in kwargs
    else:
        assert kwargs["extra_body"] == expected_extra_body


@pytest.mark.parametrize(
    ("explicit", "api_base", "expected"),
    [
        # 显式值优先，不被 host 推断覆盖
        ("vllm", "https://dashscope.aliyuncs.com/v1", "vllm"),
        ("dashscope", "http://localhost:8000/v1", "dashscope"),
        ("deepseek", "http://localhost:8000/v1", "deepseek"),
        # None 时按 host 子串匹配（大小写不敏感），无匹配回退 vllm
        (None, "https://dashscope.aliyuncs.com/v1", "dashscope"),
        (None, "https://api.dashscope.com/v1", "dashscope"),
        (None, "http://localhost:8000/v1", "vllm"),
        (None, "http://10.0.0.5:8001", "vllm"),
        (None, "HTTPS://DASHSCOPE.ALIYUNCS.COM/v1", "dashscope"),
        (None, "https://api.deepseek.com/v1", "deepseek"),
        (None, "https://API.DEEPSEEK.COM/v1", "deepseek"),
    ],
)
def test_resolve_backend_kind(
    explicit: BackendKind | None, api_base: str, expected: BackendKind
) -> None:
    assert resolve_backend_kind(explicit, api_base) == expected


# ---------- DeepSeek V4 thinking mode 矩阵 ----------


@pytest.mark.parametrize(
    ("overrides", "expected_extra_body"),
    [
        pytest.param(
            {"thinking_mode": "enabled"},
            {"thinking": {"type": "enabled"}},
            id="enabled",
        ),
        pytest.param(
            {"thinking_mode": "enabled", "reasoning_effort": "max"},
            {"thinking": {"type": "enabled", "reasoning_effort": "max"}},
            id="enabled-with-effort",
        ),
        pytest.param(
            {"thinking_mode": "disabled"},
            {"thinking": {"type": "disabled"}},
            id="disabled",
        ),
        # thinking_mode="disabled" 时 reasoning_effort 不写入 thinking 对象
        pytest.param(
            {"thinking_mode": "disabled", "reasoning_effort": "max"},
            {"thinking": {"type": "disabled"}},
            id="disabled-ignores-effort",
        ),
        # thinking_mode=None 且 enable_thinking=None → 不下发 extra_body
        pytest.param({}, None, id="no-thinking-mode-no-extra-body"),
        # 若同时设置了 thinking_mode 和 enable_thinking，thinking_mode 优先
        pytest.param(
            {"thinking_mode": "enabled", "enable_thinking": True},
            {"thinking": {"type": "enabled"}},
            id="precedence-over-enable-thinking",
        ),
    ],
)
def test_openai_adapter_deepseek_thinking_matrix(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    expected_extra_body: dict[str, Any] | None,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = _adapter(
        OpenAIModelAdapter,
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com/v1",
        **overrides,
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    if expected_extra_body is None:
        assert "extra_body" not in kwargs
    else:
        assert kwargs["extra_body"] == expected_extra_body


def test_native_adapter_deepseek_thinking_merges_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NativeToolsOpenAIAdapter: thinking + top_k 在 extra_body 里共存。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _adapter(
        NativeToolsOpenAIAdapter,
        api_base="https://api.deepseek.com/v1",
        thinking_mode="enabled",
        reasoning_effort="high",
        top_k=20,
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {
        "thinking": {"type": "enabled", "reasoning_effort": "high"},
        "top_k": 20,
    }
