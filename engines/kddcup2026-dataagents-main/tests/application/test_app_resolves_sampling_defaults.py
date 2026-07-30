"""`application.py:_resolve_sampling_defaults` 的解析测试。

覆盖三个 mode 分支：
- enable_thinking=True  → 填 Qwen3 thinking 兜底（0.6 / 0.95 / 20 / 0 / 0）
- enable_thinking=False → 填 non-thinking 兜底（0.7 / 0.8 / 20 / 0 / 1.5）
- enable_thinking=None  → 全部保持 None（与 "不下发" 语义对偶）

以及显式值优先：用户显式值（含显式 0.0 与 seed）不会被默认值覆盖，无论哪种 mode。
"""

from __future__ import annotations

import pytest

from agents.application import _resolve_sampling_defaults
from agents.config import AgentConfig


def _agent(**overrides: object) -> AgentConfig:
    """构造 AgentConfig，可选覆盖任意字段。"""
    return AgentConfig(api_key="k", **overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("enable_thinking", "expected"),
    [
        pytest.param(
            True,
            {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "seed": None,
            },
            id="thinking",
        ),
        pytest.param(
            False,
            {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 1.5,
                "seed": None,
            },
            id="non-thinking",
        ),
        # mode 未确定时不能瞎填——保持与 enable_thinking=None → 不下发的语义对偶
        pytest.param(
            None,
            {
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "min_p": None,
                "presence_penalty": None,
                "seed": None,
            },
            id="mode-unset",
        ),
    ],
)
def test_mode_fills_official_recommendation(
    enable_thinking: bool | None, expected: dict[str, object]
) -> None:
    resolved = _resolve_sampling_defaults(_agent(enable_thinking=enable_thinking))
    assert resolved == expected


@pytest.mark.parametrize(
    ("overrides", "expected_subset"),
    [
        # 显式 temperature=0.3：thinking 模式下保留 0.3，不被 0.6 覆盖；未显式字段仍走兜底
        pytest.param(
            {"enable_thinking": True, "temperature": 0.3, "top_p": 0.5},
            {
                "temperature": 0.3,
                "top_p": 0.5,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
            },
            id="thinking-explicit-override",
        ),
        # 显式 0.0 / 40 不被 non-thinking 兜底（1.5 / 20）覆盖
        pytest.param(
            {"enable_thinking": False, "presence_penalty": 0.0, "top_k": 40},
            {
                "presence_penalty": 0.0,
                "top_k": 40,
                "temperature": 0.7,
                "top_p": 0.8,
            },
            id="non-thinking-explicit-override",
        ),
        # seed 永远透传，与默认无关
        pytest.param({"enable_thinking": True, "seed": 42}, {"seed": 42}, id="seed-thinking"),
        # enable_thinking=None 时 seed 显式值仍透传，其它字段保持 None
        pytest.param(
            {"enable_thinking": None, "seed": 7},
            {"seed": 7, "temperature": None, "top_p": None},
            id="seed-none-mode",
        ),
    ],
)
def test_explicit_values_survive_mode_defaults(
    overrides: dict[str, object], expected_subset: dict[str, object]
) -> None:
    resolved = _resolve_sampling_defaults(_agent(**overrides))
    for key, value in expected_subset.items():
        assert resolved[key] == value, key
