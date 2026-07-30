"""OpenAI-compatible model adapters and response parsing."""

from __future__ import annotations

import abc
import json
import logging
import re
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from openai import APIError, OpenAI

from agents.llm.types import (
    ModelMessage,
    ModelResponse,
    ModelToolCall,
    TokenUsage,
    ToolSchemaSource,
)

if TYPE_CHECKING:
    from agents.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class _ToolCallFunctionDump(TypedDict, total=False):
    name: str
    arguments: str | dict[str, Any]


class _ToolCallDump(TypedDict, total=False):
    id: str
    type: str
    function: _ToolCallFunctionDump


# 用于 defensively 剥离 `<think>...</think>`（vLLM `--reasoning-parser qwen3` 未启用时的兜底）
_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _strip_think_tags(text: str) -> str:
    """把潜在的 `<think>...</think>` 块剔除，返回干净的外部可见文本。

    即便 vLLM 启用了 `--reasoning-parser qwen3`，少数旧版本或并非 Qwen 的后端仍可能
    把思考段塞进 `message.content`，污染正常输出，这里做一层防御性处理。
    """
    return _THINK_TAG_PATTERN.sub("", text).strip()


# 后端类型字面量。决定思考模式参数在 extra_body 中的位置：
# - "vllm":      enable_thinking 嵌套到 `chat_template_kwargs`
# - "dashscope": enable_thinking 走顶层
# - "deepseek":  thinking 对象走顶层（DeepSeek V4 API 协议，与 Qwen3 不同）
BackendKind = Literal["vllm", "dashscope", "deepseek"]

# 推断 dashscope / deepseek 用的 host 关键字。
_DASHSCOPE_HOST_HINTS = ("aliyuncs.com", "dashscope", "xf-yun.com")
_DEEPSEEK_HOST_HINTS = ("api.deepseek.com",)


def resolve_backend_kind(backend_kind: BackendKind | None, api_base: str) -> BackendKind:
    """显式 `backend_kind` 优先；为 None 时按 `api_base` 主机名推断。

    `api_base` 已在 adapter 构造时 `rstrip("/")`，可直接 lower 后做子串匹配。
    """
    if backend_kind is not None:
        return backend_kind
    lowered = api_base.lower()
    if any(hint in lowered for hint in _DEEPSEEK_HOST_HINTS):
        return "deepseek"
    if any(hint in lowered for hint in _DASHSCOPE_HOST_HINTS):
        return "dashscope"
    return "vllm"


def _thinking_extra_body(enable_thinking: bool, backend_kind: BackendKind) -> dict[str, Any]:
    """按 backend_kind 把 `enable_thinking` 包成正确的 extra_body 片段（Qwen3 路径）。

    返回值用于 dict 合并。两条路径产出**不同 key**，互斥：
    - vLLM:      `{"chat_template_kwargs": {"enable_thinking": <bool>}}`
    - DashScope: `{"enable_thinking": <bool>}`
    """
    if backend_kind == "vllm":
        return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    return {"enable_thinking": enable_thinking}


def _deepseek_thinking_extra_body(
    thinking_mode: str,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """构造 DeepSeek V4 的 thinking extra_body 片段。

    DeepSeek V4 API 协议：
    - `thinking.type`: "enabled" | "disabled"
    - `thinking.reasoning_effort`: "high" | "max"（仅 type="enabled" 时有意义）
    """
    thinking: dict[str, str] = {"type": thinking_mode}
    if reasoning_effort is not None and thinking_mode == "enabled":
        thinking["reasoning_effort"] = reasoning_effort
    return {"thinking": thinking}


def _message_to_payload(message: ModelMessage) -> dict[str, Any]:
    """把 `ModelMessage` 序列化成 OpenAI SDK 期望的 dict。

    规则：
    - tool 角色必须带 tool_call_id
    - assistant 角色若带 tool_calls，则 content 可为空串
    - 其他角色只保留 role+content

    content 可以是 str 或 list[dict]（含 video_url block）。video_url 是
    DashScope / Qwen-VL 扩展，非 OpenAI 标准；换后端时需适配。
    """
    if message.role == "tool":
        content = message.content
        if not isinstance(content, str):
            logger.warning("tool message content is not str, serializing to JSON")
            content = json.dumps(content, ensure_ascii=False)
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": message.tool_call_id or "",
        }
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "assistant" and message.tool_calls:
        payload["tool_calls"] = message.tool_calls
    return payload


def _extract_reasoning_content(message: Any) -> str:
    """从 `choices[0].message` 里防御性提取 `reasoning_content`。

    vLLM `--reasoning-parser qwen3` 与百炼 Qwen3 均会把思维链塞进这个字段。
    OpenAI 官方 SDK 的 pydantic 模型不一定声明该字段，兜底走 `model_extra`。
    Some vLLM builds expose the same field as `reasoning`.
    """
    direct = getattr(message, "reasoning_content", None)
    if isinstance(direct, str) and direct:
        return direct
    direct_reasoning = getattr(message, "reasoning", None)
    if isinstance(direct_reasoning, str) and direct_reasoning:
        return direct_reasoning
    extras = getattr(message, "model_extra", None)
    if isinstance(extras, dict):
        extra_dict = cast(dict[str, Any], extras)
        value = extra_dict.get("reasoning_content") or extra_dict.get("reasoning")
        if isinstance(value, str):
            return value
    return ""


def _as_int(value: Any) -> int:
    """把 usage 里的数值字段强制转 int，None/非法值归零。

    百炼偶尔返回字符串数字；SDK 基本保证 int，但做一层保护成本微乎其微。
    """
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_usage(response: Any) -> TokenUsage:
    """从 `chat.completions` 响应构造 `TokenUsage`。

    对嵌套字段全部走 `getattr(..., None)`，不同后端返回的字段集不一致，缺失一律视为 0。

    缓存 token 兼容两种格式：
    - OpenAI / vLLM / DashScope: `prompt_tokens_details.cached_tokens`
    - DeepSeek V4: `prompt_cache_hit_tokens`（顶层直接暴露）
    两者取 max 保证无论哪种后端都能正确提取。
    """
    usage_obj = getattr(response, "usage", None)
    if usage_obj is None:
        return TokenUsage()

    prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
    completion_details = getattr(usage_obj, "completion_tokens_details", None)
    cached = getattr(prompt_details, "cached_tokens", None) if prompt_details is not None else None
    reasoning = (
        getattr(completion_details, "reasoning_tokens", None)
        if completion_details is not None
        else None
    )

    # DeepSeek V4 returns prompt_cache_hit_tokens at top level instead of nested
    ds_cached = getattr(usage_obj, "prompt_cache_hit_tokens", None)
    resolved_cached = max(_as_int(cached), _as_int(ds_cached))

    return TokenUsage(
        prompt_tokens=_as_int(getattr(usage_obj, "prompt_tokens", 0)),
        completion_tokens=_as_int(getattr(usage_obj, "completion_tokens", 0)),
        total_tokens=_as_int(getattr(usage_obj, "total_tokens", 0)),
        cached_tokens=resolved_cached,
        reasoning_tokens=_as_int(reasoning),
    )


class ToolCallParseError(RuntimeError):
    """tool_call arguments JSON 解析失败，携带部分响应供 trace 保留。"""

    def __init__(self, message: str, *, partial_response: ModelResponse) -> None:
        super().__init__(message)
        self.partial_response = partial_response


class _BaseOpenAIAdapter(abc.ABC):
    """Shared init / request-build / call-and-parse for OpenAI-compatible adapters."""

    @abc.abstractmethod
    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: ToolSchemaSource | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        raise NotImplementedError

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float | None = None,
        max_tokens: int = 0,
        enable_thinking: bool | None = None,
        backend_kind: BackendKind | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        thinking_mode: str | None = None,
        reasoning_effort: str | None = None,
        max_retries: int = 2,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_mode = thinking_mode
        self.reasoning_effort = reasoning_effort
        self.backend_kind: BackendKind = resolve_backend_kind(backend_kind, self.api_base)
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def estimate_prompt_tokens(
        self, messages: list[ModelMessage], *, tools: ToolSchemaSource | None = None
    ) -> int:
        """Estimate request tokens; no tool schema is sent on this path."""
        del tools
        from agents.llm.rate_limit import (
            estimate_prompt_tokens as estimate_message_prompt_tokens,
        )

        return estimate_message_prompt_tokens(messages)

    def _build_request_kwargs(self, messages: list[ModelMessage]) -> dict[str, Any]:
        """Assemble the common chat.completions request dict."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_payload(message) for message in messages],
        }
        if self.temperature is not None:
            request_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            request_kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            request_kwargs["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            request_kwargs["seed"] = self.seed
        if self.max_tokens > 0:
            request_kwargs["max_tokens"] = self.max_tokens
        extra_body: dict[str, Any] = {}
        if self.thinking_mode is not None:
            extra_body.update(
                _deepseek_thinking_extra_body(self.thinking_mode, self.reasoning_effort)
            )
        elif self.enable_thinking is not None:
            extra_body.update(_thinking_extra_body(self.enable_thinking, self.backend_kind))
        if self.top_k is not None:
            extra_body["top_k"] = self.top_k
        if self.min_p is not None:
            extra_body["min_p"] = self.min_p
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        return request_kwargs

    def _call_and_parse_base(
        self, request_kwargs: dict[str, Any]
    ) -> tuple[ModelResponse, list[_ToolCallDump]]:
        """Execute a streaming API call, accumulate chunks, and return the result.

        Returns (base_response, raw_tool_call_dicts) so subclasses can do
        protocol-specific post-processing on tool calls.
        """
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        request_kwargs["stream"] = True
        request_kwargs["stream_options"] = {"include_usage": True}

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage = TokenUsage()

        started = perf_counter()
        try:
            stream = self._client.chat.completions.create(**request_kwargs)  # pyright: ignore[reportUnknownVariableType]
            for chunk in cast(list[Any], stream):
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = _extract_usage(chunk)

                choices: list[Any] = getattr(chunk, "choices", None) or []
                if not choices:
                    continue

                delta: Any = choices[0].delta

                if getattr(delta, "content", None):
                    content_parts.append(str(delta.content))

                rc = _extract_reasoning_content(delta)
                if rc:
                    reasoning_parts.append(rc)

                delta_tool_calls: list[Any] | None = getattr(delta, "tool_calls", None)
                if delta_tool_calls:
                    for tc in delta_tool_calls:
                        idx: int = tc.index
                        if idx not in tool_calls_acc:
                            fn: Any = getattr(tc, "function", None)
                            tool_calls_acc[idx] = {
                                "id": str(getattr(tc, "id", "") or ""),
                                "type": str(getattr(tc, "type", "function") or "function"),
                                "function": {
                                    "name": (str(getattr(fn, "name", "") or "")) if fn else "",
                                    "arguments": (str(getattr(fn, "arguments", "") or ""))
                                    if fn
                                    else "",
                                },
                            }
                        else:
                            entry = tool_calls_acc[idx]
                            if getattr(tc, "id", None):
                                entry["id"] = str(tc.id)
                            fn = getattr(tc, "function", None)
                            if fn:
                                if getattr(fn, "name", None):
                                    entry["function"]["name"] = str(fn.name)
                                if getattr(fn, "arguments", None):
                                    entry["function"]["arguments"] += str(fn.arguments)
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc
        latency_ms = int((perf_counter() - started) * 1000)

        content = "".join(content_parts)
        cleaned = _strip_think_tags(content)
        raw_tool_calls: list[_ToolCallDump] = [
            cast(_ToolCallDump, tool_calls_acc[i]) for i in sorted(tool_calls_acc)
        ]

        base_response = ModelResponse(
            content=cleaned,
            tool_calls=[],
            raw_response=cleaned,
            raw_tool_calls=cast(list[dict[str, Any]], raw_tool_calls),
            reasoning_content="".join(reasoning_parts),
            usage=usage,
            latency_ms=latency_ms,
        )
        return base_response, raw_tool_calls


class OpenAIModelAdapter(_BaseOpenAIAdapter):
    """对 OpenAI 兼容 chat.completions 接口的包装（无工具）。

    仅依赖 `openai` 官方 SDK；通过 `api_base` 切换后端（本地 vLLM、代理网关等）。
    不传 `tools=` 参数，模型响应为纯文本。
    """

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: ToolSchemaSource | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """发起一次 chat.completions 请求并把响应打包成 `ModelResponse`。

        非 native 工具协议：请求体没有 tools 字段，``tools`` 参数接受但忽略。
        ``response_format`` 透传给 API（如 ``{"type": "json_object"}``）。
        """
        del tools, kwargs
        request_kwargs = self._build_request_kwargs(messages)
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        base_response, _ = self._call_and_parse_base(request_kwargs)
        return base_response


class NativeToolsOpenAIAdapter(_BaseOpenAIAdapter):
    """对 OpenAI 兼容接口的包装（原生 function calling 协议）。

    与 `OpenAIModelAdapter` 的核心差异：
    - 请求端：传 `tools=[...]` + `tool_choice`（默认 "auto"，可传 named-function dict
      强制调用指定工具）+ `parallel_tool_calls=True`，
      把工具定义下推给模型，让其原生以 tool_calls 形式回应
    - 响应端：解析 `choices[0].message.tool_calls`，映射为 `ModelToolCall` 列表，
      `content` 保留为 assistant 的自由文本（可能为空字符串）

    vLLM 侧需启用 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`。
    """

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float | None = None,
        tools: ToolRegistry,
        allow_parallel_tool_calls: bool = True,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int = 0,
        enable_thinking: bool | None = None,
        backend_kind: BackendKind | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        thinking_mode: str | None = None,
        reasoning_effort: str | None = None,
        max_retries: int = 2,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            backend_kind=backend_kind,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            seed=seed,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            max_retries=max_retries,
            timeout=timeout,
        )
        self.tools = tools
        self.allow_parallel_tool_calls = allow_parallel_tool_calls
        # None → "auto"（现行为）；video 子代理传 named-function dict 强制指向 report
        self.tool_choice: str | dict[str, Any] = tool_choice if tool_choice is not None else "auto"
        # 预先把构造期 ToolRegistry 的 definitions 转成 OpenAI tools 数组并缓存：
        # registry 不可变，缓存在适配器生命周期内永远有效，保证默认路径（tools=None
        # 或同一 registry 对象）重复请求的 tools 字段字节稳定（prompt cache 命中）。
        # per-call override 现场渲染、不缓存：`to_openai_tools` 按名排序输出确定，
        # 字节稳定不依赖缓存；身份缓存有 CPython id 复用陷阱，不值得。
        self._openai_tools = tools.to_openai_tools()
        self._tools_prompt_tokens: int | None = None

    def _resolve_openai_tools(self, tools: ToolSchemaSource | None) -> list[dict[str, Any]]:
        """Resolve the tools payload: default registry hits the cache, overrides render fresh."""
        if tools is None or tools is self.tools:
            return self._openai_tools
        return tools.to_openai_tools()

    def _estimate_tools_prompt_tokens(self) -> int:
        """Count the stable native tools schema prompt contribution lazily."""
        if self._tools_prompt_tokens is None:
            from agents.llm.tokenizer import count_qwen_tokens

            tools_text = json.dumps(self._openai_tools, ensure_ascii=False)
            self._tools_prompt_tokens = count_qwen_tokens(tools_text)
        return self._tools_prompt_tokens

    def estimate_prompt_tokens(
        self, messages: list[ModelMessage], *, tools: ToolSchemaSource | None = None
    ) -> int:
        """Estimate native request tokens, including vLLM-rendered tool definitions."""
        if tools is None or tools is self.tools:
            tools_tokens = self._estimate_tools_prompt_tokens()
        else:
            from agents.llm.tokenizer import count_qwen_tokens

            tools_tokens = count_qwen_tokens(
                json.dumps(tools.to_openai_tools(), ensure_ascii=False)
            )
        return super().estimate_prompt_tokens(messages) + tools_tokens

    def _build_request_kwargs(
        self, messages: list[ModelMessage], *, tools: ToolSchemaSource | None = None
    ) -> dict[str, Any]:
        request_kwargs = super()._build_request_kwargs(messages)
        request_kwargs["tools"] = self._resolve_openai_tools(tools)
        request_kwargs["tool_choice"] = self.tool_choice
        if self.allow_parallel_tool_calls:
            request_kwargs["parallel_tool_calls"] = True
        return request_kwargs

    def complete(
        self,
        messages: list[ModelMessage],
        *,
        tools: ToolSchemaSource | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """发起一次 chat.completions 请求，返回解析好的 `ModelResponse`。"""
        request_kwargs = self._build_request_kwargs(messages, tools=tools)
        base_response, raw_tool_calls = self._call_and_parse_base(request_kwargs)

        parsed_calls: list[ModelToolCall] = []

        for dumped in raw_tool_calls:
            function_payload: _ToolCallFunctionDump = dumped.get("function") or {}
            name = str(function_payload.get("name", ""))
            arguments_raw = function_payload.get("arguments", "")
            arguments: dict[str, Any]
            if isinstance(arguments_raw, dict):
                arguments = dict(arguments_raw)
            else:
                arg_str = str(arguments_raw or "").strip()
                if not arg_str:
                    arguments = {}
                else:
                    try:
                        parsed = json.loads(arg_str)
                    except json.JSONDecodeError as exc:
                        err = ToolCallParseError(
                            f"Failed to parse tool_call arguments for {name!r}: {exc}",
                            partial_response=ModelResponse(
                                content=base_response.content,
                                tool_calls=[],
                                raw_response=base_response.raw_response,
                                raw_tool_calls=base_response.raw_tool_calls,
                                reasoning_content=base_response.reasoning_content,
                                usage=base_response.usage,
                                latency_ms=base_response.latency_ms,
                            ),
                        )
                        raise err from exc
                    if not isinstance(parsed, dict):
                        raise RuntimeError(
                            f"tool_call arguments for {name!r} must decode to an object."
                        )
                    arguments = cast(dict[str, Any], parsed)

            call_id = str(dumped.get("id") or "")
            parsed_calls.append(ModelToolCall(id=call_id, name=name, arguments=arguments))

        return ModelResponse(
            content=base_response.content,
            tool_calls=parsed_calls,
            raw_response=base_response.raw_response,
            raw_tool_calls=base_response.raw_tool_calls,
            reasoning_content=base_response.reasoning_content,
            usage=base_response.usage,
            latency_ms=base_response.latency_ms,
        )
