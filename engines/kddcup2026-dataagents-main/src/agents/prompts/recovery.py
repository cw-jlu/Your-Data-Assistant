"""Observation prompt rendering and recovery hints."""

from __future__ import annotations

import json
from typing import cast


def _recovery_hint(observation: dict[str, object]) -> str | None:
    """根据 observation 中的 error 信息生成定向恢复提示。"""
    error_text = str(observation.get("error", "")).lower()
    raw_content = observation.get("content")
    if isinstance(raw_content, dict):
        inner = str(cast(dict[str, object], raw_content).get("error", ""))
        error_text = f"{error_text} {inner.lower()}"
    if not error_text.strip():
        return None
    if "no such column" in error_text or "no such table" in error_text:
        return "Re-check schema via preview_file on the .db/.sqlite file."
    if "syntax error" in error_text:
        return "Fix the syntax and retry."
    if "timeout" in error_text or "timed out" in error_text:
        return "Simplify: use SQL aggregation or add WHERE."
    if "mergeerror" in error_text or "merge on" in error_text:
        return (
            "Merge/join key dtype mismatch. Cast both sides to str before merging: "
            "df1['key'] = df1['key'].astype(str); df2['key'] = df2['key'].astype(str)."
        )
    if "keyerror" in error_text or "not in index" in error_text:
        return "Column name not found. Print df.columns.tolist() to verify exact names."
    if "no such file" in error_text or "filenotfounderror" in error_text:
        return (
            "File not found. Use relative paths from the context directory (CWD) or "
            "os.path.join(context_root, 'subdir/file.csv'). Never hard-code absolute paths."
        )
    if "failed to parse tool_call arguments for 'answer'" in error_text:
        return (
            "The answer tool_call arguments contained malformed JSON — "
            "inline columns/rows is too large or malformed to serialize reliably. "
            "Switch to the artifact path: in execute_python, write the answer "
            "DataFrame to os.path.join(os.environ['DABENCH_ANSWER_DIR'], 'answer.csv') "
            "via df.to_csv(..., index=False), then call "
            'answer({"from_csv": "<that absolute path>"}).'
        )
    return "Review the error and adjust."


def build_observation_prompt(observation: dict[str, object], *, ok: bool | None = None) -> str:
    """把工具执行结果以格式化 JSON 回传给模型作为下一轮的观察。

    ensure_ascii=False 保留中文等非 ASCII 字符原样可读；indent=2 让模型更易 parse。
    当 ok=False 时追加定向恢复提示帮助模型纠偏。
    """
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    result = f"Observation:\n{rendered}"
    if ok is False:
        hint = _recovery_hint(observation)
        if hint:
            result += f"\n\nRecovery hint: {hint}"
    return result
