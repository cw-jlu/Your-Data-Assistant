from __future__ import annotations

from typing import Any

from data_agent_baseline.token_utils import count_tokens, truncate_by_tokens

TRUNCATION_SUFFIX = (
    "\n\n...（内容已被截断，返回过长。"
    "如未获取到所需信息，请尝试其他方式而非直接读取全部文档。）"
)


def truncate_str(text: str, max_tokens: int = 10000) -> str:
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    return truncate_by_tokens(text, max_tokens) + TRUNCATION_SUFFIX


def truncate_content(
    obj: Any,
    *,
    max_str_tokens: int = 2000,
    max_list_items: int = 200,
) -> Any:
    if isinstance(obj, str):
        return truncate_str(obj, max_str_tokens)

    if isinstance(obj, list):
        truncated = obj[:max_list_items]
        result: list[Any] = [
            truncate_content(item, max_str_tokens=max_str_tokens, max_list_items=max_list_items)
            for item in truncated
        ]
        if len(obj) > max_list_items:
            result.append(TRUNCATION_SUFFIX)
        return result

    if isinstance(obj, dict):
        return {
            key: truncate_content(value, max_str_tokens=max_str_tokens, max_list_items=max_list_items)
            for key, value in obj.items()
        }

    return obj


def truncate_answer_content(
    answer: dict[str, Any],
    *,
    max_str_tokens: int = 2000,
    max_list_items: int = 200,
) -> dict[str, Any]:
    """Truncate answer rows for context while preserving answer column names."""
    truncated: dict[str, Any] = {}
    if "columns" in answer:
        columns = answer.get("columns")
        truncated["columns"] = list(columns) if isinstance(columns, list) else columns
    if "rows" in answer:
        truncated["rows"] = truncate_content(
            answer.get("rows"),
            max_str_tokens=max_str_tokens,
            max_list_items=max_list_items,
        )
    for key, value in answer.items():
        if key not in truncated:
            truncated[key] = truncate_content(
                value,
                max_str_tokens=max_str_tokens,
                max_list_items=max_list_items,
            )
    return truncated
