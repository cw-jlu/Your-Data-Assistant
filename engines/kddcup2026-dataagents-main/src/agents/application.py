"""App 工厂：把 dataset/registry/model adapter/agent 的 wiring 收敛到一处。

`runner.py` 不再直接持有具体的 `OpenAIModelAdapter` / `ToolRegistry` / `ReActAgent`
依赖；它只调用 `build_application(config)` 拿到 `AgentApp`，再用 `app.new_agent()` 驱动循环。

线上路径
    runner 子进程在 key 固定后调 `build_application(pinned_config)`，构造完整 AgentApp。
    `AgentApp` 不会被 pickle 跨进程——只有 `AppConfig` 跨进程，`AgentApp` 始终在子进程内构造。

测试路径
    pytest 通过 `build_application(config, model=ScriptedModelAdapter(...), tools=mock_registry)`
    跳过真实 OpenAI 客户端构造；override 完全保留原有 model/tools 注入语义。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from agents.agent import ReActAgent, ReActAgentConfig
from agents.benchmark.dataset import DABenchPublicDataset
from agents.config import AgentConfig, AppConfig, reject_placeholder_api_key
from agents.llm import (
    ModelAdapter,
    NativeToolsOpenAIAdapter,
    OpenAIModelAdapter,
    RateLimitedAdapter,
)
from agents.llm.openai import resolve_backend_kind
from agents.runtime import StepCallback
from agents.tools import ToolRegistry, create_default_tool_registry, create_run_etl_tool_definition
from agents.tools.explore import create_explore_tool_definition
from agents.tools.explore_video import create_explore_video_tool_definition
from agents.verification import AnswerVerifier

# Qwen3 model card Best Practices 推荐的两套采样组合：
# - thinking 模式：温度低、top_p 高、不抑制 presence
# - non-thinking 模式：温度更高、top_p 收紧、用 presence_penalty 抑制复读
# `seed` 在两套里都不下发（保留服务端默认行为）。
_THINKING_DEFAULTS: dict[str, float | int] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
}
_NON_THINKING_DEFAULTS: dict[str, float | int] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
}
_SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "seed",
)
_ANSWER_VERIFIER_MAX_TOKENS = 512


class _Sentinel(Enum):
    MISSING = "MISSING"


_SENTINEL = _Sentinel.MISSING


def _resolve_sampling_defaults(
    agent_config: AgentConfig,
    *,
    enable_thinking_override: bool | None | _Sentinel = _SENTINEL,
) -> dict[str, Any]:
    """按思考模式给未显式设置的采样字段填推荐兜底。

    Qwen3 路径（enable_thinking）：
    - True  → _THINKING_DEFAULTS
    - False → _NON_THINKING_DEFAULTS
    - None  → 不填，全部 None 透传

    DeepSeek 路径（thinking_mode）：
    - "enabled"  → 清除全部采样参数，服务端忽略
    - "disabled" → 仅清除 seed（DeepSeek API 不支持 seed）
    - None       → 不填

    `seed` 在 Qwen3 路径保持用户原始意图（None=不下发），但 DeepSeek 不支持。

    *enable_thinking_override* — 非 _SENTINEL 时替代 ``agent_config.enable_thinking``
    用于选择 defaults 表，但仍从 agent_config 读用户显式值。

    返回值是 ``{field: resolved_value_or_None}``，调用方逐字段下发到 Adapter。
    """
    resolved: dict[str, Any] = {field: getattr(agent_config, field) for field in _SAMPLING_FIELDS}

    if agent_config.thinking_mode is not None:
        if agent_config.thinking_mode == "enabled":
            for field in _SAMPLING_FIELDS:
                resolved[field] = None
        else:
            resolved["seed"] = None
        return resolved

    effective = (
        enable_thinking_override
        if enable_thinking_override is not _SENTINEL
        else agent_config.enable_thinking
    )
    if effective is None:
        return resolved

    defaults = _THINKING_DEFAULTS if effective else _NON_THINKING_DEFAULTS
    for field, default_value in defaults.items():
        if resolved[field] is None:
            resolved[field] = default_value
    return resolved


def _has_rate_limiter(config: AppConfig) -> bool:
    """Return whether model calls for this pinned config should use the rate limiter."""
    return config.agent.rate_limit is not None and bool(config.agent.api_key)


def _maybe_wrap_rate_limiter(
    config: AppConfig,
    inner: ModelAdapter,
    *,
    reserve_output_tokens: int | None = None,
) -> ModelAdapter:
    """Apply the shared rate-limit wrapper to any adapter that calls the model API."""
    rate_limit = config.agent.rate_limit
    if rate_limit is None or not config.agent.api_key:
        return inner
    if reserve_output_tokens is not None:
        rate_limit = replace(rate_limit, reserve_output_tokens=reserve_output_tokens)
    return RateLimitedAdapter(
        inner=inner,
        api_key=config.agent.api_key,
        rate_limit=rate_limit,
    )


def build_model_adapter(config: AppConfig, tools: ToolRegistry) -> ModelAdapter:
    """根据 `config.agent` 构造模型适配器。

    始终构造 `NativeToolsOpenAIAdapter`，需要传入 `tools` 以把工具 schema
    下推到 OpenAI `tools=[...]` 字段。

    当 `config.agent.rate_limit` 非 None 且 `api_key` 非空时，外面再套一层
    `RateLimitedAdapter`（PR-2）。线上 vLLM 路径留空 rate_limit 即跳过。
    """
    sampling = _resolve_sampling_defaults(config.agent)
    reject_placeholder_api_key(config.agent.api_key, field="agent.api_key")

    has_rate_limiter = _has_rate_limiter(config)

    inner = NativeToolsOpenAIAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        tools=tools,
        allow_parallel_tool_calls=config.agent.allow_parallel_tool_calls,
        max_tokens=config.agent.max_tokens,
        enable_thinking=config.agent.enable_thinking,
        thinking_mode=config.agent.thinking_mode,
        reasoning_effort=config.agent.reasoning_effort,
        backend_kind=config.agent.backend_kind,
        max_retries=0 if has_rate_limiter else 2,
        **sampling,
    )

    return _maybe_wrap_rate_limiter(config, inner)


def build_no_tools_model_adapter(config: AppConfig) -> ModelAdapter:
    """Build a plain chat-completion adapter for verifier-style sub-agents.

    This deliberately never attaches tool schemas. It still uses the shared
    rate limiter so verifier calls consume the configured RPM/TPM budget.
    """
    reject_placeholder_api_key(config.agent.api_key, field="agent.api_key")
    has_rate_limiter = _has_rate_limiter(config)
    resolved_kind = resolve_backend_kind(config.agent.backend_kind, config.agent.api_base)
    is_deepseek = config.agent.thinking_mode is not None or resolved_kind == "deepseek"
    inner = OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        min_p=0.0,
        presence_penalty=0.0,
        max_tokens=_ANSWER_VERIFIER_MAX_TOKENS,
        enable_thinking=None if is_deepseek else False,
        thinking_mode="disabled" if is_deepseek else None,
        backend_kind=config.agent.backend_kind,
        seed=config.agent.seed,
        max_retries=0 if has_rate_limiter else 2,
    )
    return _maybe_wrap_rate_limiter(config, inner)


def _build_explorer_adapter(config: AppConfig) -> ModelAdapter:
    """Build the single model adapter shared by all Explorer runs.

    Uses the same model/API/sampling config as the main agent. The
    constructor registry is the plain (no-video) explorer toolset and only
    serves bare ``complete()`` calls as a fallback; each `run_explorer` run
    assembles its own registry (optionally with `explore_video`) and pushes
    it per call, so advertisement and dispatch share one instance.

    Rate limiting is shared: all adapters use the same ``api_key`` →
    ``FileTokenBucket`` automatically deduplicates on the same bucket file.
    """
    sampling = _resolve_sampling_defaults(config.agent)
    reject_placeholder_api_key(config.agent.api_key, field="agent.api_key")
    has_rate_limiter = _has_rate_limiter(config)

    from agents.explorer.registry import create_explorer_tool_registry

    inner = NativeToolsOpenAIAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        tools=create_explorer_tool_registry(),
        allow_parallel_tool_calls=config.agent.allow_parallel_tool_calls,
        max_tokens=config.agent.max_tokens,
        enable_thinking=config.agent.enable_thinking,
        thinking_mode=config.agent.thinking_mode,
        reasoning_effort=config.agent.reasoning_effort,
        backend_kind=config.agent.backend_kind,
        max_retries=0 if has_rate_limiter else 2,
        **sampling,
    )

    return _maybe_wrap_rate_limiter(config, inner)


def _build_video_adapter(config: AppConfig) -> ModelAdapter:
    """Build the report-only video sub-agent adapter."""
    reject_placeholder_api_key(config.agent.api_key, field="agent.api_key")
    has_rate_limiter = _has_rate_limiter(config)
    max_tokens = (
        config.agent.video_max_tokens
        if config.agent.video_max_tokens is not None
        else config.agent.max_tokens
    )

    video_enable_thinking = (
        config.agent.video_enable_thinking
        if config.agent.video_enable_thinking is not None
        else config.agent.enable_thinking
    )
    sampling = _resolve_sampling_defaults(
        config.agent,
        enable_thinking_override=video_enable_thinking,
    )

    from agents.video.registry import create_video_tool_registry

    tool_choice: str | dict[str, Any] = "auto"
    if config.agent.video_tool_choice == "forced":
        tool_choice = {"type": "function", "function": {"name": "report"}}

    inner = NativeToolsOpenAIAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        tools=create_video_tool_registry(),
        allow_parallel_tool_calls=False,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        enable_thinking=video_enable_thinking,
        thinking_mode=config.agent.thinking_mode,
        reasoning_effort=config.agent.reasoning_effort,
        backend_kind=config.agent.backend_kind,
        max_retries=0 if has_rate_limiter else 2,
        **sampling,
    )

    reserve_output_tokens = None
    if (
        config.agent.video_max_tokens is not None
        and max_tokens > 0
        and (config.agent.max_tokens <= 0 or max_tokens > config.agent.max_tokens)
    ):
        reserve_output_tokens = max_tokens

    return _maybe_wrap_rate_limiter(
        config,
        inner,
        reserve_output_tokens=reserve_output_tokens,
    )


def build_answer_verifier(config: AppConfig) -> AnswerVerifier:
    """Build the no-tools answer verifier used by online application wiring."""
    return AnswerVerifier(model=build_no_tools_model_adapter(config), max_rejections=2)


@dataclass(frozen=True, slots=True)
class AgentApp:
    """ReAct 循环运行所需的全部依赖集合。

    字段都是构造完毕的实例：
    - dataset:        从 `config.dataset.root_path` 加载的只读 task 集合
    - registry:       不可变工具注册表（默认 8 工具）
    - model_adapter:  按 rate_limit 包装好的模型适配器
    - answer_verifier: 提交前 answer 形态检查器；测试 override 路径可为 None
    - agent_config:   ReAct 循环预算（max_steps / empty retry）

    `new_agent` 每次产出一个独立 `ReActAgent` 实例：共享 model/tools/config，
    但循环状态相互隔离，便于同一进程串行跑多个任务。
    """

    dataset: DABenchPublicDataset
    registry: ToolRegistry
    model_adapter: ModelAdapter
    answer_verifier: AnswerVerifier | None
    agent_config: ReActAgentConfig

    def new_agent(self, step_callback: StepCallback | None = None) -> ReActAgent:
        """生成一个使用本 App 共享依赖的新 ReAct 实例。"""
        return ReActAgent(
            model=self.model_adapter,
            tools=self.registry,
            config=self.agent_config,
            step_callback=step_callback,
            answer_verifier=self.answer_verifier,
        )


def build_application(
    config: AppConfig,
    *,
    model: ModelAdapter | None = None,
    tools: ToolRegistry | None = None,
    answer_verifier: AnswerVerifier | None = None,
) -> AgentApp:
    """从 `AppConfig` 一次性构造 ReAct 运行所需的全部依赖。

    `model` / `tools` / `answer_verifier` 是给测试预留的 override：
    - 传 `model=ScriptedModelAdapter(...)` 跳过真实 OpenAI 客户端构造
    - 传 `tools=mock_registry`（或一个裁剪过的 ToolRegistry）跳过默认 8 工具
    - 传 `answer_verifier=...` 可覆盖线上默认 verifier；`model` override 默认不建 verifier

    线上路径 override 参数都为 None，走完整的默认 wiring。

    注意：`AgentApp` 实例不可 pickle（`OpenAI` 客户端含线程锁），所以本函数必须在
    子进程内调用——主进程只跨进程传递 `AppConfig`。
    """
    dataset = DABenchPublicDataset(config.dataset.root_path)

    model_ref: dict[str, ModelAdapter] | None = None
    if tools is not None:
        registry = tools
    else:
        model_ref = {}

        def get_main_model() -> ModelAdapter:
            assert model_ref is not None
            try:
                return model_ref["model"]
            except KeyError as exc:
                raise RuntimeError(
                    "run_etl requested before model adapter was initialized"
                ) from exc

        video_adapter = _build_video_adapter(config)
        video_tool = create_explore_video_tool_definition(video_adapter)
        explorer_adapter = _build_explorer_adapter(config)
        explore_def = create_explore_tool_definition(explorer_adapter, video_tool=video_tool)
        run_etl_def = create_run_etl_tool_definition(get_main_model)
        registry = create_default_tool_registry(explore_tool=explore_def, run_etl_tool=run_etl_def)

    model_adapter = model if model is not None else build_model_adapter(config, tools=registry)
    if model_ref is not None:
        model_ref["model"] = model_adapter
    verifier = answer_verifier
    if verifier is None and model is None and config.agent.enable_answer_verifier:
        verifier = build_answer_verifier(config)
    agent_config = ReActAgentConfig(
        max_steps=config.agent.max_steps,
        max_empty_tool_call_retries=config.agent.max_empty_tool_call_retries,
        enable_preact=config.agent.enable_preact,
    )
    return AgentApp(
        dataset=dataset,
        registry=registry,
        model_adapter=model_adapter,
        answer_verifier=verifier,
        agent_config=agent_config,
    )


__all__ = [
    "AgentApp",
    "build_answer_verifier",
    "build_application",
    "build_model_adapter",
    "build_no_tools_model_adapter",
]

# `_resolve_sampling_defaults` 故意不进 __all__：保持私有 API，但允许测试 import。
