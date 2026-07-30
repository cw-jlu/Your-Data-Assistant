"""Behavior tests for terminal-policy producing-code extraction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agents.runtime.state import AgentRuntimeState
from agents.verification.terminal_policy import check_unit_conversions, recent_execute_code


def _state(*steps: tuple[str, dict[str, Any]]) -> AgentRuntimeState:
    state = AgentRuntimeState()
    state.steps.extend(
        SimpleNamespace(action=action, action_input=action_input)  # type: ignore[arg-type]
        for action, action_input in steps
    )
    return state


def test_recent_execute_code_returns_last_execute_step_only() -> None:
    state = _state(
        ("execute_python", {"code": "first = 1"}),
        ("execute_context_sql", {"sql": "SELECT 1"}),
        ("read_csv", {"path": "a.csv"}),
        ("execute_python", {"code": "final = build_answer()"}),
    )

    code = recent_execute_code(state)

    assert code is not None
    assert "final = build_answer()" in code
    assert "SELECT 1" not in code
    assert "first = 1" not in code


def test_recent_execute_code_skips_empty_inputs_and_handles_absence() -> None:
    assert recent_execute_code(None) is None
    assert recent_execute_code(_state(("read_csv", {"path": "a.csv"}))) is None

    state = _state(
        ("execute_python", {"code": "real = 1"}),
        ("execute_python", {"code": "   "}),
    )
    code = recent_execute_code(state)
    assert code is not None
    assert "real = 1" in code


def test_check_unit_conversions_matches_answer_columns_case_insensitively(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setattr("agents.config.ETL_SCRATCH_ROOT", scratch)
    cache_dir = scratch / "task_1" / "_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "sample_units.json").write_text(
        '{"amount": "元", "_target_AMOUNT": "万元", "_factor_AMOUNT": "0.0001"}',
        encoding="utf-8",
    )

    message = check_unit_conversions(
        SimpleNamespace(task_id="task_1"),  # type: ignore[arg-type]
        ["Amount"],
        [[500000]],
    )

    assert message is not None
    assert "×0.0001" in message
