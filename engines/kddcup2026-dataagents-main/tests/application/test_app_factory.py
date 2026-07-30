"""`agents.application.build_application` 工厂单测。

覆盖点：
- `build_application(config)` 返回 `AgentApp`，字段全部赋值且类型正确
- `new_agent()` 每次调用返回独立的 `ReActAgent` 实例（不共享循环状态）
- `agent_config` 透传 `config.agent.max_steps` / `max_empty_tool_call_retries`
- no-tools adapter 的真实选择逻辑（plain OpenAI / rate-limit wrapper）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents.agent import ReActAgent, ReActAgentConfig
from agents.application import AgentApp, build_application, build_no_tools_model_adapter
from agents.benchmark.dataset import DABenchPublicDataset
from agents.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    RateLimitConfig,
    RunConfig,
)
from agents.llm.openai import NativeToolsOpenAIAdapter, OpenAIModelAdapter
from agents.llm.rate_limit import RateLimitedAdapter
from agents.tools.registry import ToolRegistry, create_default_tool_registry
from agents.verification.answer import AnswerVerifier


def _make_config(
    *,
    tmp_path: Path,
    api_key: str = "primary-key",
    max_steps: int = 13,
    max_empty_tool_call_retries: int = 5,
    rate_limit: RateLimitConfig | None = None,
) -> AppConfig:
    """构造一个最小可用的 AppConfig，避免真实数据集依赖。"""
    return AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "data"),
        agent=AgentConfig(
            model="m",
            api_base="http://x",
            api_key=api_key,
            max_steps=max_steps,
            max_empty_tool_call_retries=max_empty_tool_call_retries,
            rate_limit=rate_limit,
        ),
        run=RunConfig(
            output_dir=tmp_path / "runs",
            run_id=None,
            max_workers=1,
            task_timeout_seconds=0,
        ),
    )


def test_build_application_populates_all_fields(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path=tmp_path)
    app = build_application(cfg)

    assert isinstance(app, AgentApp)
    assert isinstance(app.dataset, DABenchPublicDataset)
    assert app.dataset.root_dir == cfg.dataset.root_path
    assert isinstance(app.registry, ToolRegistry)
    assert {"answer", "execute_python", "execute_context_sql"} <= set(app.registry.definitions)
    assert isinstance(app.agent_config, ReActAgentConfig)
    assert app.agent_config.max_steps == 13
    assert app.agent_config.max_empty_tool_call_retries == 5
    assert hasattr(app.model_adapter, "complete")
    assert isinstance(app.answer_verifier, AnswerVerifier)


def test_build_application_rejects_placeholder_api_key(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path=tmp_path, api_key="YOUR_API_KEY")

    with pytest.raises(ValueError, match=r"agent\.api_key.*placeholder"):
        build_application(cfg)


def test_new_agent_returns_fresh_instance_per_call(tmp_path: Path) -> None:
    """每次 new_agent() 都是一个独立的 ReActAgent，循环状态不共享。"""
    cfg = _make_config(tmp_path=tmp_path)
    app = build_application(cfg)

    agent_a = app.new_agent()
    agent_b = app.new_agent()

    assert isinstance(agent_a, ReActAgent)
    assert isinstance(agent_b, ReActAgent)
    assert agent_a is not agent_b
    assert agent_a.model is agent_b.model
    assert agent_a.tools is agent_b.tools
    assert agent_a.config is agent_b.config
    assert agent_a.answer_verifier is agent_b.answer_verifier


def test_build_no_tools_model_adapter_uses_plain_openai_adapter(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path=tmp_path)

    adapter = build_no_tools_model_adapter(cfg)

    assert isinstance(adapter, OpenAIModelAdapter)
    assert adapter.enable_thinking is False
    assert not hasattr(adapter, "_openai_tools")


def test_build_no_tools_model_adapter_preserves_rate_limit_wrapper(tmp_path: Path) -> None:
    cfg = _make_config(
        tmp_path=tmp_path,
        rate_limit=RateLimitConfig(state_dir=tmp_path / "rl"),
    )

    adapter = build_no_tools_model_adapter(cfg)

    assert isinstance(adapter, RateLimitedAdapter)
    assert isinstance(adapter.inner, OpenAIModelAdapter)
    assert adapter.inner.enable_thinking is False


def test_build_application_tools_override(tmp_path: Path) -> None:
    """传 tools=custom_registry 时跳过 create_default_tool_registry。"""
    cfg = _make_config(tmp_path=tmp_path)
    custom = create_default_tool_registry()
    app = build_application(cfg, tools=custom)
    assert app.registry is custom


def _video_cfg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    **agent_overrides: Any,
) -> AppConfig:
    from unittest.mock import MagicMock

    from agents.llm import openai as model_mod

    monkeypatch.setattr(model_mod, "OpenAI", MagicMock())
    agent = AgentConfig(api_key="test-key", **agent_overrides)
    return AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "data"),
        agent=agent,
        run=RunConfig(output_dir=tmp_path / "runs"),
    )


def test_build_video_adapter_forced_tool_choice_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agents.application import _build_video_adapter

    adapter = _build_video_adapter(_video_cfg(monkeypatch, tmp_path, video_tool_choice="forced"))
    assert isinstance(adapter, NativeToolsOpenAIAdapter)
    assert adapter.tool_choice == {"type": "function", "function": {"name": "report"}}
    assert list(adapter.tools.definitions) == ["report"]
    assert adapter.allow_parallel_tool_calls is False


def test_build_video_adapter_default_auto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agents.application import _build_video_adapter

    adapter = _build_video_adapter(_video_cfg(monkeypatch, tmp_path))
    assert isinstance(adapter, NativeToolsOpenAIAdapter)
    assert adapter.tool_choice == "auto"


def test_build_video_adapter_nothink_override_uses_low_temperature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agents.application import _build_video_adapter

    adapter = _build_video_adapter(
        _video_cfg(monkeypatch, tmp_path, enable_thinking=True, video_enable_thinking=False)
    )
    assert isinstance(adapter, NativeToolsOpenAIAdapter)
    assert adapter.enable_thinking is False
    assert adapter.temperature == 0.7


def test_build_video_adapter_thinking_override_keeps_thinking_temperature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agents.application import _build_video_adapter

    adapter = _build_video_adapter(
        _video_cfg(monkeypatch, tmp_path, enable_thinking=False, video_enable_thinking=True)
    )
    assert isinstance(adapter, NativeToolsOpenAIAdapter)
    assert adapter.enable_thinking is True
    assert adapter.temperature == 0.6


def test_build_video_adapter_uses_video_max_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agents.application import _build_video_adapter

    adapter = _build_video_adapter(
        _video_cfg(monkeypatch, tmp_path, max_tokens=8192, video_max_tokens=16384)
    )
    assert isinstance(adapter, NativeToolsOpenAIAdapter)
    assert adapter.max_tokens == 16384


def test_build_video_adapter_rate_limit_reserves_video_max_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agents.application import _build_video_adapter

    adapter = _build_video_adapter(
        _video_cfg(
            monkeypatch,
            tmp_path,
            max_tokens=8192,
            video_max_tokens=16384,
            rate_limit=RateLimitConfig(
                reserve_output_tokens=8192,
                state_dir=tmp_path / "rl",
            ),
        )
    )
    assert isinstance(adapter, RateLimitedAdapter)
    assert adapter.rate_limit_config.reserve_output_tokens == 16384
    assert isinstance(adapter.inner, NativeToolsOpenAIAdapter)
    assert adapter.inner.max_tokens == 16384
