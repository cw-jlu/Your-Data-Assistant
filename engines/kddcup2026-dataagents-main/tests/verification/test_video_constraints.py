"""Behavior tests for the deterministic video-constraint post-check.

Covers:
- extraction of batch_id / snapshot_dates from explore_video and from nested
  explore(video_findings) observations
- batch_id-ignored detection (year-LIKE without batch reference)
- snapshot-multiplicity detection (multi-snapshot collapsed to 1 row)
- aggregate-keyword exemption (英中混合)
- conservative no-op cases (missing constraints / state / code)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agents.runtime.state import AgentRuntimeState
from agents.verification.video_constraints import (
    VideoConstraints,
    check_batch_id_filter,
    check_snapshot_multiplicity,
    evaluate_video_constraints,
    extract_video_constraints,
)


def _state(*steps: tuple[str, dict[str, Any]]) -> AgentRuntimeState:
    state = AgentRuntimeState()
    state.steps.extend(
        SimpleNamespace(action=action, observation=observation)  # type: ignore[arg-type]
        for action, observation in steps
    )
    return state


def _video_obs(
    batch_id: str | None = None, snapshot_dates: list[str] | None = None
) -> dict[str, Any]:
    rules: dict[str, Any] = {}
    if batch_id is not None:
        rules["batch_id"] = batch_id
    if snapshot_dates is not None:
        rules["snapshot_dates"] = snapshot_dates
    return {
        "ok": True,
        "tool": "explore_video",
        "content": {
            "video_status": "ok",
            "findings": {"extracted_data": {"rules": rules}},
        },
    }


# ---------- extract_video_constraints -----------------------------------------


def test_extracts_batch_id_and_snapshots_from_explore_video() -> None:
    state = _state(
        (
            "explore_video",
            _video_obs(batch_id="B-2004", snapshot_dates=["2004-03-31", "2004-12-31"]),
        )
    )
    c = extract_video_constraints(state)
    assert c is not None
    assert c.batch_id == "B-2004"
    assert c.snapshot_dates == ("2004-03-31", "2004-12-31")


def test_extracts_from_nested_explore_video_findings() -> None:
    """主 explore 工具把每次 explore_video 的完整 content dict 塞进 video_findings list。

    生产结构是 ``video_findings = [{video_status, path, findings: <dict>, ...}, ...]``,
    必须再剥一层 .findings 才能拿到 extracted_data —— 这条钉子防止以后误改回扁平结构。
    """
    state = _state(
        (
            "explore",
            {
                "ok": True,
                "tool": "explore",
                "content": {
                    "video_findings": [
                        {
                            "video_status": "ok",
                            "path": "video/demo.mp4",
                            "findings": {
                                "extracted_data": {"rules": {"batch_id": "BATCH-2021-Q4"}}
                            },
                            "video_steps_used": 3,
                        }
                    ]
                },
            },
        )
    )
    c = extract_video_constraints(state)
    assert c is not None
    assert c.batch_id == "BATCH-2021-Q4"


def test_extracts_from_nested_explore_video_findings_legacy_flat() -> None:
    """老格式兜底:video_findings list 元素就是 findings dict 自身(单测 / 老测试用)。"""
    state = _state(
        (
            "explore",
            {
                "ok": True,
                "tool": "explore",
                "content": {
                    "video_findings": [{"extracted_data": {"rules": {"batch_id": "OLD-2020"}}}]
                },
            },
        )
    )
    c = extract_video_constraints(state)
    assert c is not None
    assert c.batch_id == "OLD-2020"


def test_falls_back_to_statistical_period_when_not_plain_year() -> None:
    """少数视频把 batch 串塞进 statistical_period — 但 4 位纯年份不算批次串。"""
    state = _state(
        (
            "explore_video",
            {
                "ok": True,
                "tool": "explore_video",
                "content": {
                    "findings": {
                        "extracted_data": {"rules": {"statistical_period": "BATCH-2024-Q3-007"}}
                    }
                },
            },
        )
    )
    c = extract_video_constraints(state)
    assert c is not None
    assert c.batch_id == "BATCH-2024-Q3-007"


def test_pure_year_statistical_period_is_not_a_batch_id() -> None:
    """`statistical_period: "2021"` 不是批次串,不应被当作 batch_id。"""
    state = _state(
        (
            "explore_video",
            {
                "ok": True,
                "tool": "explore_video",
                "content": {
                    "findings": {"extracted_data": {"rules": {"statistical_period": "2021"}}}
                },
            },
        )
    )
    assert extract_video_constraints(state) is None


def test_picks_latest_findings_when_explore_video_called_twice() -> None:
    state = _state(
        ("explore_video", _video_obs(batch_id="OLD-2020-Q1")),
        ("execute_python", {"ok": True, "tool": "execute_python", "content": {}}),
        ("explore_video", _video_obs(batch_id="NEW-2021-Q4")),
    )
    c = extract_video_constraints(state)
    assert c is not None
    assert c.batch_id == "NEW-2021-Q4"


def test_returns_none_when_no_video_step_present() -> None:
    state = _state(("execute_python", {"ok": True, "tool": "execute_python", "content": {}}))
    assert extract_video_constraints(state) is None


def test_returns_none_when_state_is_none() -> None:
    assert extract_video_constraints(None) is None


# ---------- check_batch_id_filter ---------------------------------------------


def test_batch_filter_rejects_year_like_when_batch_id_absent() -> None:
    """task_31 回归:视频说 BATCH-2021-Q4,SQL 写 `endate LIKE '2021%'` 必须拦。"""
    constraints = VideoConstraints(batch_id="BATCH-2021-Q4", snapshot_dates=())
    code = "[execute_context_sql]\nSELECT * FROM mf_x WHERE endate LIKE '2021%'"
    msg = check_batch_id_filter(constraints, code)
    assert msg is not None
    assert "video_batch_ignored" in msg
    assert "BATCH-2021-Q4" in msg


def test_batch_filter_accepts_when_full_batch_id_present_in_code() -> None:
    constraints = VideoConstraints(batch_id="BATCH-2021-Q4", snapshot_dates=())
    code = "[execute_context_sql]\nSELECT * FROM mf_x WHERE batch_id = 'BATCH-2021-Q4'"
    assert check_batch_id_filter(constraints, code) is None


def test_batch_filter_accepts_when_quarter_marker_present() -> None:
    """code 没写完整 batch_id 但用了对应季度,认为已经收窄到对的子周期。"""
    constraints = VideoConstraints(batch_id="BATCH-2021-Q4", snapshot_dates=())
    code = "[execute_python]\ndf_q4 = df[(df['year']==2021) & (df['quarter']=='Q4')]"
    assert check_batch_id_filter(constraints, code) is None


def test_batch_filter_accepts_b_2004_with_year_only_code() -> None:
    """B-2004 没有季度后缀,代码用 2004 年份就已经是最具体的子周期 —— 不拦。"""
    constraints = VideoConstraints(batch_id="B-2004", snapshot_dates=())
    code = "[execute_python]\ndf_2004 = df[df['report_date'].str.startswith('2004')]"
    assert check_batch_id_filter(constraints, code) is None


def test_batch_filter_silent_when_year_absent_in_code() -> None:
    """code 完全不带年份(比如对全表 count) —— batch 检查也没什么可拦的。"""
    constraints = VideoConstraints(batch_id="BATCH-2021-Q4", snapshot_dates=())
    code = "[execute_python]\ncount = len(df[df['performance'] < 0])"
    assert check_batch_id_filter(constraints, code) is None


def test_batch_filter_ignores_batch_id_mentions_inside_comments() -> None:
    """task_31 回归:agent 把 batch_id 抄进 # 注释但实际 filter 用泛年份。

    剥注释前 checker 会被注释里的 BATCH-2021-Q4 / Q4 字面量骗到放行。
    剥注释后,看到执行代码只有 startswith('2021'),应当 reject。
    """
    constraints = VideoConstraints(batch_id="BATCH-2021-Q4", snapshot_dates=())
    code = (
        "[execute_python]\n"
        "import pandas as pd\n"
        "# BATCH-2021-Q4 表示 2021年第四季度\n"
        "# 筛选endate为2021年Q4的数据（10月-12月）\n"
        "df_2021 = df[df['endate'].str.startswith('2021')]\n"
    )
    msg = check_batch_id_filter(constraints, code)
    assert msg is not None
    assert "video_batch_ignored" in msg


def test_batch_filter_accepts_sql_block_comment_then_real_quarter_filter() -> None:
    """SQL 块注释也要剥;但代码里真的有 Q4 filter 还是要放行。"""
    constraints = VideoConstraints(batch_id="BATCH-2021-Q4", snapshot_dates=())
    code = (
        "/* batch BATCH-2021-Q4: filter for Q4 quarter only */\n"
        "SELECT * FROM x WHERE year = 2021 AND quarter = 'Q4'"
    )
    assert check_batch_id_filter(constraints, code) is None


def test_batch_filter_silent_on_missing_inputs() -> None:
    assert check_batch_id_filter(None, "SELECT 1") is None
    assert (
        check_batch_id_filter(VideoConstraints(batch_id="BATCH-2021-Q4", snapshot_dates=()), None)
        is None
    )


# ---------- check_snapshot_multiplicity ---------------------------------------


def test_snapshot_check_rejects_single_row_for_multi_snapshot_batch() -> None:
    """task_35 回归:3 个 snapshot_dates,答案 1 行,问题不要求聚合 → 拦。"""
    constraints = VideoConstraints(
        batch_id="B-2004",
        snapshot_dates=("2004-03-31", "2004-06-30", "2004-12-31"),
    )
    question = "what are the corporate deposit and fiscal deposit amounts for that batch year?"
    msg = check_snapshot_multiplicity(constraints, question, row_count=1)
    assert msg is not None
    assert "snapshot_multiplicity" in msg
    assert "3" in msg


def test_snapshot_check_accepts_when_question_asks_for_aggregate() -> None:
    constraints = VideoConstraints(
        batch_id="B-2004",
        snapshot_dates=("2004-03-31", "2004-06-30", "2004-12-31"),
    )
    for question in (
        "what is the TOTAL deposit across the batch?",
        "average deposits for batch B-2004",
        "汇总该批次的存款数",
        "求该批次的平均存款",
    ):
        assert check_snapshot_multiplicity(constraints, question, row_count=1) is None


def test_snapshot_check_silent_when_row_count_matches_snapshots() -> None:
    constraints = VideoConstraints(
        batch_id="B-2004",
        snapshot_dates=("2004-03-31", "2004-06-30", "2004-12-31"),
    )
    assert check_snapshot_multiplicity(constraints, "what are the values?", row_count=3) is None


def test_snapshot_check_silent_when_only_one_snapshot() -> None:
    constraints = VideoConstraints(batch_id=None, snapshot_dates=("2024-06-30",))
    assert check_snapshot_multiplicity(constraints, "what is the value?", row_count=1) is None


# ---------- evaluate_video_constraints (integration) --------------------------


def test_evaluate_returns_batch_warning_when_batch_ignored() -> None:
    state = _state(("explore_video", _video_obs(batch_id="BATCH-2021-Q4")))
    result = evaluate_video_constraints(
        state,
        "top 10 funds by profit",
        row_count=10,
        producing_code="SELECT * FROM x WHERE endate LIKE '2021%'",
    )
    assert result is not None
    assert result.warning is not None
    assert "video_batch_ignored" in result.warning
    assert result.rejection is None


def test_evaluate_returns_snapshot_rejection_when_collapsed() -> None:
    state = _state(
        (
            "explore_video",
            _video_obs(
                batch_id="B-2004", snapshot_dates=["2004-03-31", "2004-06-30", "2004-12-31"]
            ),
        )
    )
    result = evaluate_video_constraints(
        state,
        "what are the deposit amounts?",
        row_count=1,
        producing_code="df_2004 = df[df['report_date'].str.startswith('2004')]",
    )
    assert result is not None
    assert result.rejection is not None
    assert "snapshot_multiplicity" in result.rejection
    assert result.warning is None


def test_evaluate_returns_none_when_no_video_constraints() -> None:
    state = _state(("execute_python", {"ok": True, "tool": "execute_python", "content": {}}))
    assert (
        evaluate_video_constraints(state, "question", row_count=1, producing_code="SELECT 1")
        is None
    )


def test_evaluate_uses_latest_execute_step_from_state_when_no_code_passed() -> None:
    """task_31 回归:agent 中间一步做对(Q4 filter),最后一步退回泛年份。

    evaluate 必须只看最新那一步的 code,不要被早期 Q4 变量名/字符串字面量蒙骗。
    """
    state = AgentRuntimeState()
    state.steps.append(
        SimpleNamespace(  # type: ignore[arg-type]
            action="explore_video",
            observation=_video_obs(batch_id="BATCH-2021-Q4"),
            action_input={},
        )
    )
    state.steps.append(
        SimpleNamespace(  # type: ignore[arg-type]
            action="execute_python",
            observation={"ok": True, "tool": "execute_python", "content": {}},
            action_input={
                "code": (
                    "df_q4 = df[df['endate'].str.startswith('2021-1')]\n"
                    "print('Q4 数据行数:', len(df_q4))"
                )
            },
        )
    )
    state.steps.append(
        SimpleNamespace(  # type: ignore[arg-type]
            action="execute_python",
            observation={"ok": True, "tool": "execute_python", "content": {}},
            action_input={
                "code": "df_2021 = df[df['endate'].str.startswith('2021')]\nresult = df_2021.head(10)"
            },
        )
    )
    result = evaluate_video_constraints(state, "top 10 funds", row_count=10)
    assert result is not None
    assert result.warning is not None
    assert "video_batch_ignored" in result.warning
    assert result.rejection is None
