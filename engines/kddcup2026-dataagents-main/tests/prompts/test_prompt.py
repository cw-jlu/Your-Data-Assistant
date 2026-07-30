from __future__ import annotations

from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.prompts import (
    REACT_NATIVE_SYSTEM_PROMPT,
    build_native_system_prompt,
    build_observation_prompt,
    build_task_prompt,
)


def _task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="Find the requested value.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_react_native_system_prompt_constant_matches_default_builder() -> None:
    """模块级常量 `REACT_NATIVE_SYSTEM_PROMPT` 必须等于 `build_native_system_prompt()`。

    向后兼容钉子：早期 callers（notebooks、自定义 A/B 配置）直接 import 常量；
    重构后常量被 `build_native_system_prompt()` 动态生成，两者必须 byte-equal。
    """
    assert build_native_system_prompt() == REACT_NATIVE_SYSTEM_PROMPT


def test_native_system_prompt_custom_override_is_preserved() -> None:
    assert build_native_system_prompt("custom native prompt") == "custom native prompt"


def test_task_prompt_keeps_only_task_specific_context(tmp_path: Path) -> None:
    """user 消息只承载任务特定上下文：question 本身。

    跨任务的不变量（工具决策、tool_call 协议、路径约定）全部由 system prompt
    + path 字段 schema description 承担。user 消息保持极简，最大化 prompt
    cache 命中率与 `Question:` 的注意力。
    """
    assert build_task_prompt(_task(tmp_path)) == "Question: Find the requested value."


def test_system_prompts_use_runtime_clock_for_current_year() -> None:
    """`current year` 语义必须来自 execute_python 运行时，而不是数据最大日期。"""
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "current year" in lower
    assert "execute_python" in prompt
    assert "date.today()" in prompt
    assert "not" in lower and "latest date in the dataset" in lower


def test_system_prompt_distinguishes_video_rules_from_displayed_samples() -> None:
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "video_findings" in prompt
    assert "findings.extracted_data.rules" in prompt
    assert "findings.extracted_data.displayed_samples" in prompt
    assert "never copy" in lower
    assert "screen previews/examples only" in lower
    assert "preview_file" in prompt
    assert "df.columns" in prompt


def test_system_prompt_empty_result_recovery_preserves_question_conditions() -> None:
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "never submit an empty result" in lower
    assert "re-check column/value mapping" in lower
    assert "formats, units, joins, and time grain" in lower
    assert "do not drop or weaken explicit question conditions" in lower
    assert "broaden filters or relax conditions" not in lower


def test_system_prompt_observed_data_overrides_conflicting_knowledge() -> None:
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "authoritative starting point" in lower
    assert "matches observed schema/data" in lower
    assert "if sampling or computation proves a conflict" in lower
    assert "observed data wins" in lower
    assert "re-derive from evidence" in lower


def test_system_prompt_requires_on_demand_etl_after_explore() -> None:
    prompt = build_native_system_prompt()

    assert "etl_sources" in prompt
    assert "PDF entity group samples (`entity_groups_sample`)" in prompt
    assert "call `run_etl` with exactly those document paths before computing" in prompt
    assert "csv_files[*].csv_path" in prompt
    assert "Do NOT ETL documents that are not listed" in prompt


def test_system_prompt_classifies_data_listing_triggers() -> None:
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "data-listing trigger" in lower
    assert "查/看/找/搜/显示/展示/列出" in prompt
    assert "数据/记录/明细/情况/是多少" in prompt
    assert "return source rows at original grain" in lower
    assert "do not require every requested column to be non-null" in lower
    assert "非空/不为空/有数据/non-null/not empty" in prompt


def test_sqlite_index_prompt_matches_sqlite_plan_output() -> None:
    prompt = build_native_system_prompt()

    assert "SCAN <table>" in prompt
    assert "SCAN TABLE" not in prompt
    assert "idx_<table>_<col>" in prompt
    assert "idx_<col>" not in prompt


def test_system_prompt_keeps_sqlite_guidance_domain_general() -> None:
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "encounter scoping" not in lower
    assert "admission/discharge" not in lower
    assert "stay id" not in lower


# ---------- build_observation_prompt: ok parameter & recovery hints ----------


@pytest.mark.parametrize(
    ("error", "ok", "expected_hint"),
    [
        pytest.param("no such column: foo", None, None, id="ok-none-no-hint"),
        pytest.param("no such column: foo", True, None, id="ok-true-no-hint"),
        pytest.param("no such column: amount", False, "preview_file", id="no-such-column"),
        pytest.param("no such table: orders", False, "preview_file", id="no-such-table"),
        pytest.param('near "SELEC": syntax error', False, "Fix the syntax", id="syntax-error"),
        pytest.param("execution timed out after 30s", False, "Simplify", id="timeout"),
        pytest.param(
            "model.complete failed: Failed to parse tool_call arguments for 'answer': bad json",
            False,
            "DABENCH_ANSWER_DIR",
            id="answer-json-parse-error",
        ),
        pytest.param("some unexpected failure", False, "Review the error", id="generic-error"),
    ],
)
def test_observation_prompt_recovery_hints(
    error: str, ok: bool | None, expected_hint: str | None
) -> None:
    """ok=False 时按 error 签名路由 recovery hint；ok=None/True 不追加。"""
    result = build_observation_prompt({"error": error}, ok=ok)

    if expected_hint is None:
        assert "Recovery hint" not in result
    else:
        assert "Recovery hint:" in result
        assert expected_hint in result
