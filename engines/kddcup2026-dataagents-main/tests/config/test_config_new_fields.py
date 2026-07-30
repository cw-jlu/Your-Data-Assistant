"""`config.load_app_config` 对新加字段的解析测试。

覆盖：
- 全字段缺省默认值（单测吸收所有 `*_default_is_*` 断言）
- unknown field 拒收（load 路径 + 构造器路径）
- 非法值拒收（数值边界 / 枚举 / 互斥组合）的参数化表
- 非平凡归一化（大小写、0 语义、explicit null、组合字段）
- thinking + greedy decoding 的 UserWarning 触发矩阵
- tracing 块解析（缺省 None / enabled 默认 db_path / 显式 db_path）
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agents.config import AgentConfig, RunConfig, load_app_config


def _write_config(path: Path, body: str) -> Path:
    """写一个最小可解析 YAML；body 会被插入到 agent: 下面作为额外字段。"""
    path.write_text(f"agent:\n  model: m\n  api_base: http://x\n  api_key: k\n{body}")
    return path


def test_defaults_when_fields_absent(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "")
    cfg = load_app_config(cfg_path)

    assert cfg.agent.max_tokens == 16384
    assert cfg.agent.video_max_tokens is None
    assert cfg.agent.max_empty_tool_call_retries == 8
    assert cfg.agent.enable_thinking is None
    # backend_kind 缺省 → None，由 adapter 按 api_base 自动推断
    assert cfg.agent.backend_kind is None
    # 旧默认 temperature=0.0 已废弃；mode-aware 兜底由 _resolve_sampling_defaults 完成
    assert cfg.agent.temperature is None
    assert cfg.agent.top_p is None
    assert cfg.agent.seed is None
    assert cfg.agent.thinking_mode is None
    assert cfg.agent.reasoning_effort is None
    assert cfg.agent.enable_preact is False
    assert cfg.tracing is None


def test_load_app_config_rejects_unknown_agent_field(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  does_not_exist: 123\n")
    with pytest.raises(ValueError, match="Unexpected keyword argument"):
        load_app_config(cfg_path)


def test_config_constructors_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="does_not_exist"):
        AgentConfig(does_not_exist=123)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="does_not_exist"):
        RunConfig(does_not_exist=123)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param(
            "  max_empty_tool_call_retries: -1\n",
            "max_empty_tool_call_retries",
            id="empty-retries-negative",
        ),
        pytest.param("  max_tokens: -1\n", "max_tokens", id="max-tokens-negative"),
        pytest.param(
            "  video_max_tokens: -1\n", "video_max_tokens", id="video-max-tokens-negative"
        ),
        # `enable_thinking: 1` 不应被静默当作 True —— 明确报错，避免运维踩坑
        pytest.param("  enable_thinking: 1\n", "enable_thinking", id="enable-thinking-non-bool"),
        pytest.param("  backend_kind: ollama\n", "backend_kind", id="backend-kind-unknown"),
        pytest.param("  temperature: -0.1\n", "temperature", id="temperature-negative"),
        pytest.param("  top_p: 0.0\n", "top_p", id="top-p-zero"),
        pytest.param("  top_p: 1.1\n", "top_p", id="top-p-above-one"),
        pytest.param("  top_k: 0\n", "top_k", id="top-k-zero"),
        pytest.param("  top_k: -1\n", "top_k", id="top-k-negative"),
        pytest.param("  min_p: 1.5\n", "min_p", id="min-p-above-one"),
        pytest.param("  min_p: -0.1\n", "min_p", id="min-p-negative"),
        pytest.param("  presence_penalty: 2.5\n", "presence_penalty", id="presence-above-two"),
        pytest.param(
            "  presence_penalty: -2.5\n", "presence_penalty", id="presence-below-minus-two"
        ),
        pytest.param(
            '  thinking_mode: "auto"\n',
            "Input should be 'enabled' or 'disabled'",
            id="thinking-mode-unknown",
        ),
        pytest.param(
            '  thinking_mode: "enabled"\n  reasoning_effort: "low"\n',
            "Input should be 'high' or 'max'",
            id="reasoning-effort-unknown",
        ),
        # enable_thinking (Qwen3) 与 thinking_mode (DeepSeek) 互斥
        pytest.param(
            '  enable_thinking: true\n  thinking_mode: "enabled"\n',
            "mutually exclusive",
            id="thinking-mutual-exclusion",
        ),
        # reasoning_effort 只能在 thinking_mode="enabled" 时设置
        pytest.param(
            '  thinking_mode: "disabled"\n  reasoning_effort: "high"\n',
            "reasoning_effort",
            id="reasoning-effort-requires-enabled",
        ),
        pytest.param(
            '  reasoning_effort: "high"\n', "reasoning_effort", id="reasoning-effort-alone"
        ),
    ],
)
def test_load_app_config_rejects_invalid_values(tmp_path: Path, body: str, match: str) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    with pytest.raises(ValueError, match=match):
        load_app_config(cfg_path)


def _dotted(cfg: Any, path: str) -> Any:
    value = cfg
    for part in path.split("."):
        value = getattr(value, part)
    return value


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # 大小写规范化，保留 strict 字段集合校验
        pytest.param(
            "  backend_kind: VLLM\n", {"agent.backend_kind": "vllm"}, id="backend-kind-case"
        ),
        # 0 = 不下发 max_tokens，是合法值
        pytest.param("  max_tokens: 0\n", {"agent.max_tokens": 0}, id="max-tokens-zero"),
        pytest.param(
            "  video_max_tokens: 16384\n",
            {"agent.video_max_tokens": 16384},
            id="video-max-tokens-explicit",
        ),
        pytest.param(
            "  video_max_tokens: 0\n",
            {"agent.video_max_tokens": 0},
            id="video-max-tokens-zero",
        ),
        # explicit null 与缺省同义（enable_thinking 是 None/True/False 三态）
        pytest.param(
            "  enable_thinking: null\n", {"agent.enable_thinking": None}, id="enable-thinking-null"
        ),
        pytest.param(
            '  thinking_mode: "enabled"\n  reasoning_effort: "max"\n',
            {"agent.thinking_mode": "enabled", "agent.reasoning_effort": "max"},
            id="thinking-mode-with-effort",
        ),
    ],
)
def test_load_app_config_normalizes_nontrivial_values(
    tmp_path: Path, body: str, expected: dict[str, Any]
) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    cfg = load_app_config(cfg_path)
    for path, value in expected.items():
        assert _dotted(cfg, path) == value, path


def test_tracing_block_parsing(tmp_path: Path) -> None:
    """tracing 块：enabled → 默认 db_path；显式 db_path → 透传。缺省 → None
    已并入 `test_defaults_when_fields_absent`。"""
    enabled = load_app_config(
        _write_config(tmp_path / "enabled.yaml", "tracing:\n  enabled: true\n")
    ).tracing
    assert enabled is not None
    assert enabled.enabled is True
    assert isinstance(enabled.db_path, Path)

    custom_path = str(tmp_path / "custom.db")
    custom = load_app_config(
        _write_config(
            tmp_path / "custom.yaml", f"tracing:\n  enabled: true\n  db_path: {custom_path}\n"
        )
    ).tracing
    assert custom is not None
    assert str(custom.db_path) == custom_path


# ---------- thinking + greedy decoding 的 UserWarning ----------


@pytest.mark.parametrize(
    ("body", "expects_warning"),
    [
        # Qwen3 model card 反推荐 thinking 模式下 greedy decoding；显式 0.0 仍允许，
        # 但加载时必须发 UserWarning，避免运维静默踩坑
        pytest.param(
            "  enable_thinking: true\n  temperature: 0.0\n", True, id="thinking-greedy-warns"
        ),
        # 未显式 temperature → mode-aware 兜底会填 0.6，不应触发 warning
        pytest.param("  enable_thinking: true\n", False, id="thinking-unset-temp-silent"),
        # non-thinking 模式不在 Qwen3 反推荐范围
        pytest.param(
            "  enable_thinking: false\n  temperature: 0.0\n", False, id="no-thinking-greedy-silent"
        ),
    ],
)
def test_greedy_decoding_warning_matrix(tmp_path: Path, body: str, expects_warning: bool) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_app_config(cfg_path)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert any("greedy" in str(w.message).lower() for w in user_warnings) is expects_warning


def test_video_tool_choice_default_auto() -> None:
    assert AgentConfig().video_tool_choice == "auto"
    assert AgentConfig(video_tool_choice="forced").video_tool_choice == "forced"
