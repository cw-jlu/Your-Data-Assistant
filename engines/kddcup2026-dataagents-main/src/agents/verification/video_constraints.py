"""Deterministic video-evidence post-checks for the answer verifier.

When `explore_video` returns `findings.extracted_data.rules` with a specific
batch identifier or a multi-date snapshot schedule, the agent's filter / answer
shape MUST honor that constraint. Two failure modes recur on video tasks:

1. ``batch_id="BATCH-2021-Q4"`` is reduced to ``WHERE endate LIKE '2021%'`` —
   the year filter sums all four quarters and corrupts ranking results.
2. ``snapshot_dates=[3 dates]`` is collapsed to a single answer row even though
   the question is not aggregating across snapshots.

Both are detectable from (a) the video findings dict stored in step
observations and (b) the producing-code text. This module does the detection
without any LLM call so it can run before the LLM verifier and never produces
false positives from model interpretation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

from agents.runtime.state import AgentRuntimeState

_EXECUTE_ACTIONS: tuple[str, ...] = ("execute_python", "execute_context_sql")

# 视频证据里"该用具体批次而不是年份"的关键字段；按优先级回退。
_BATCH_ID_KEYS: tuple[str, ...] = ("batch_id", "batch_no", "archive_id", "snapshot_id")
# 多 snapshot 字段的别名 —— video agent 偶尔用 reporting_dates / quarter_ends。
_SNAPSHOT_KEYS: tuple[str, ...] = ("snapshot_dates", "reporting_dates", "quarter_ends")
# 触发"snapshot 必须 N 行"豁免的聚合关键字（英中混合,大小写不敏感）。
_AGGREGATE_KEYWORDS: tuple[str, ...] = (
    "total",
    "average",
    "sum",
    "mean",
    "aggregate",
    "overall",
    "across periods",
    "across snapshots",
    "总",
    "平均",
    "合计",
    "汇总",
    "总计",
    "总数",
    "总额",
)

# 19/20 开头的 4 位年份；前后必须不是数字以免匹配进 ID 中段。
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
# 季度后缀（Q1-Q4），用于 batch_id 含季度时检查 producing_code 是否兜住了。
_QUARTER_RE = re.compile(r"Q[1-4]\b", re.IGNORECASE)
# Python `#` / SQL `--` 单行注释；agent 经常把视频原文(含 batch_id 和季度)
# 抄到注释里再写错的 filter,不剥注释会被注释里的字符串迷惑得失之毫厘。
_LINE_COMMENT_RE = re.compile(r"(?:#|--)[^\n]*")
# SQL 块注释 /* ... */,跨行,允许内部空白。
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(code: str) -> str:
    """Drop Python/SQL comments so substring checks reflect executable text only."""
    no_block = _BLOCK_COMMENT_RE.sub(" ", code)
    return _LINE_COMMENT_RE.sub(" ", no_block)


def _latest_execute_code(state: AgentRuntimeState | None) -> str | None:
    """Return the input of the SINGLE most recent execute step, or None.

    Different from ``terminal_policy.recent_execute_code`` (which joins the last
    two) — the batch-id check needs to inspect ONLY the code that produced the
    final answer. Earlier exploration steps often contain variable names like
    ``df_q4`` or string literals like ``"Q4 数据"`` that wrongly look like
    sub-period filters and mask a wrong year-only fallback in the last step.
    """
    if state is None:
        return None
    for step in reversed(state.steps):
        if getattr(step, "action", None) not in _EXECUTE_ACTIONS:
            continue
        payload = _as_dict(getattr(step, "action_input", None)) or {}
        code = payload.get("code") or payload.get("sql") or ""
        text = str(code).strip()
        if text:
            return text
    return None


@dataclass(frozen=True, slots=True)
class VideoConstraints:
    """Filter constraints extracted from the latest video findings step."""

    batch_id: str | None
    snapshot_dates: tuple[str, ...]


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Narrow an unknown payload to ``dict[str, Any]`` for pyright; None otherwise."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _latest_video_findings(state: AgentRuntimeState | None) -> dict[str, Any] | None:
    """Walk steps in reverse, return the most recent video findings dict.

    Picks up both direct ``explore_video`` calls and the ``video_findings`` list
    nested inside the main ``explore`` tool's output.
    """
    if state is None:
        return None
    for step in reversed(state.steps):
        action = getattr(step, "action", None)
        obs = _as_dict(getattr(step, "observation", None))
        if obs is None:
            continue
        content = _as_dict(obs.get("content"))
        if content is None:
            continue
        if action == "explore_video":
            findings = _as_dict(content.get("findings"))
            if findings is not None:
                return findings
        elif action == "explore":
            # `explore` 工具把每次 explore_video 的完整 ToolExecutionResult.content
            # (i.e. {video_status, path, findings, video_steps_used})作为元素塞进
            # video_findings 列表 —— 必须再剥一层 .findings 才能拿到 extracted_data。
            vf_raw = content.get("video_findings")
            if isinstance(vf_raw, list) and vf_raw:
                latest = _as_dict(cast(list[Any], vf_raw)[-1])
                if latest is not None:
                    inner = _as_dict(latest.get("findings"))
                    if inner is not None:
                        return inner
                    # 兜底:若直接把 findings dict 塞进了 list(老格式 / 测试用)
                    if "extracted_data" in latest:
                        return latest
            else:
                vf_dict = _as_dict(vf_raw)
                if vf_dict is not None:
                    inner = _as_dict(vf_dict.get("findings"))
                    if inner is not None:
                        return inner
                    if "extracted_data" in vf_dict:
                        return vf_dict
    return None


def extract_video_constraints(state: AgentRuntimeState | None) -> VideoConstraints | None:
    """Return batch / snapshot constraints from the latest video findings, or None."""
    findings = _latest_video_findings(state)
    if findings is None:
        return None
    extracted = _as_dict(findings.get("extracted_data"))
    if extracted is None:
        return None
    rules = _as_dict(extracted.get("rules"))
    if rules is None:
        return None

    batch_id: str | None = None
    for key in _BATCH_ID_KEYS:
        value = rules.get(key)
        if isinstance(value, str) and value.strip():
            batch_id = value.strip()
            break
    if batch_id is None:
        # 兜底:statistical_period 偶尔承载完整批次串(非纯年份时)。
        period = rules.get("statistical_period")
        if isinstance(period, str) and period.strip():
            stripped = period.strip()
            if not re.fullmatch(r"\d{4}", stripped):
                batch_id = stripped

    snapshot_dates: tuple[str, ...] = ()
    for key in _SNAPSHOT_KEYS:
        value = rules.get(key)
        if isinstance(value, list) and value:
            items = cast(list[Any], value)
            snapshot_dates = tuple(str(item) for item in items if item is not None)
            break

    if batch_id is None and not snapshot_dates:
        return None
    return VideoConstraints(batch_id=batch_id, snapshot_dates=snapshot_dates)


def check_batch_id_filter(
    constraints: VideoConstraints | None,
    producing_code: str | None,
) -> str | None:
    """Warn (non-blocking) when producing_code filters by the batch's year alone.

    Returns a warning string when ALL hold:
    1. video declared a non-year batch identifier
    2. the batch's 4-digit year appears in the code
    3. the full batch identifier literal does NOT appear
    4. if the batch carries a Qn suffix, that quarter literal also does NOT appear

    The warning is informational — the batch_id from the video configuration
    panel is often just a label, not necessarily a data filter requirement.
    Downstream callers should attach this as a hint, not a hard rejection.
    """
    if constraints is None or constraints.batch_id is None or not producing_code:
        return None
    executable = _strip_comments(producing_code)
    batch_id = constraints.batch_id
    if batch_id in executable:
        return None
    year_match = _YEAR_RE.search(batch_id)
    if not year_match:
        return None
    year = year_match.group(0)
    if year not in executable:
        return None
    sub_period = batch_id[year_match.end() :].strip("-_/ .")
    if not sub_period:
        return None
    quarter_match = _QUARTER_RE.search(batch_id)
    if quarter_match and quarter_match.group(0).upper() in executable.upper():
        return None
    return (
        "Note from video-constraint check (check=video_batch_ignored): "
        f"explore_video declared batch_id={batch_id!r}, but the producing code filters "
        f"by the year {year!r} alone. If the data has a batch column, consider filtering "
        "on it; otherwise the year-level filter may already be correct — the batch_id "
        "is sometimes just a configuration label, not a data filter."
    )


def check_snapshot_multiplicity(
    constraints: VideoConstraints | None,
    question: str,
    row_count: int,
) -> str | None:
    """Reject when a multi-snapshot batch's answer collapsed into a single row.

    Skipped when the question explicitly asks for an aggregate across snapshots
    (``total`` / ``average`` / 总 / 平均 etc.) — in that case 1 row IS the
    expected shape.
    """
    if constraints is None or len(constraints.snapshot_dates) <= 1:
        return None
    if row_count != 1:
        return None
    if any(keyword in question.lower() for keyword in _AGGREGATE_KEYWORDS):
        return None
    n = len(constraints.snapshot_dates)
    dates_preview = list(constraints.snapshot_dates)
    return (
        "ANSWER REJECTED by video-constraint check (check=snapshot_multiplicity): "
        f"explore_video declared {n} snapshot dates ({dates_preview}) for the batch, but "
        "the answer table has only 1 row. The question does not request an aggregate "
        f"across snapshots — emit one row per snapshot date ({n} rows total) using the "
        "matching source records, then resubmit `answer`."
    )


@dataclass(frozen=True, slots=True)
class VideoConstraintResult:
    """Separate hard rejections from soft warnings."""

    rejection: str | None = None
    warning: str | None = None


def evaluate_video_constraints(
    state: AgentRuntimeState | None,
    question: str,
    row_count: int,
    producing_code: str | None = None,
) -> VideoConstraintResult | None:
    """Run all deterministic video checks.

    Returns a result with separate ``rejection`` (hard block) and ``warning``
    (informational, do not block submission) fields, or None.

    ``producing_code`` is optional and accepted for testing; when not provided,
    the batch-id check pulls the LATEST single execute step from state (the
    code that actually built the submitted table).
    """
    constraints = extract_video_constraints(state)
    if constraints is None:
        return None
    code_for_batch = producing_code if producing_code is not None else _latest_execute_code(state)
    warning = check_batch_id_filter(constraints, code_for_batch)
    rejection = check_snapshot_multiplicity(constraints, question, row_count)
    if warning is None and rejection is None:
        return None
    return VideoConstraintResult(rejection=rejection, warning=warning)
