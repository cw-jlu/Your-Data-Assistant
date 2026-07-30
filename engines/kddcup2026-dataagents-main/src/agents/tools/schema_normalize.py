"""Pydantic JSON Schema → vLLM-compatible normalization."""

from __future__ import annotations

from typing import Any, cast


def normalize_for_vllm(schema: dict[str, Any]) -> dict[str, Any]:
    """把 Pydantic 生成的 JSON Schema 重写成 vLLM 端的字节稳定形态。

    三处确定性变换（与历史手写 schema 字节等价）：
    1. 递归剥掉每一层 dict 的 `title` 键（Pydantic 会在每个 property 加 `title:
       "Field Name"` 字样；xgrammar 不消费它，但保留会改请求体字节）。
    2. 把 `{"anyOf": [{"type": T}, {"type": "null"}]}`（含两种顺序）折叠为
       `{"type": [T, "null"]}`，与历史 `_READ_DOC_SCHEMA` 中 `keyword` 一致；
       兄弟键（`default`/`description` 等）原样保留。
    3. 顶层 object 缺省 `required` 字段时补 `required: []` —— Pydantic 在没有
       必填字段时会省略该键，但历史手写 schema 显式写了 `required: []`。

    其它任何结构原样不动；该函数纯函数式，输入对象不被改写。
    """

    def _walk(node: Any) -> Any:
        if isinstance(node, list):
            return [_walk(item) for item in cast(list[Any], node)]
        if not isinstance(node, dict):
            return node
        original = cast(dict[str, Any], node)
        # 先做 anyOf-nullable 折叠：要在剥 title 之前判断结构，避免子元素被先改写。
        any_of_value = original.get("anyOf")
        if isinstance(any_of_value, list) and len(cast(list[Any], any_of_value)) == 2:
            entries = cast(list[Any], any_of_value)
            non_null_type: str | None = None
            null_seen = False
            ok = True
            for entry in entries:
                if not isinstance(entry, dict):
                    ok = False
                    break
                entry_dict = cast(dict[str, Any], entry)
                # 仅识别"形如单 type 的简形 anyOf 分支"，含其他键则跳过折叠
                if set(entry_dict.keys()) != {"type"}:
                    ok = False
                    break
                inner_type = entry_dict["type"]
                if not isinstance(inner_type, str):
                    ok = False
                    break
                if inner_type == "null":
                    null_seen = True
                else:
                    non_null_type = inner_type
            if ok and null_seen and non_null_type is not None:
                rewritten: dict[str, Any] = {}
                for key, value in original.items():
                    if key == "anyOf":
                        continue
                    if key == "title":
                        continue
                    rewritten[key] = _walk(value)
                rewritten["type"] = [non_null_type, "null"]
                return rewritten

        result: dict[str, Any] = {}
        for key, value in original.items():
            if key == "title":
                continue
            result[key] = _walk(value)
        if result.get("type") == "object" and "required" not in result:
            result["required"] = []
        return result

    walked = _walk(schema)
    return cast(dict[str, Any], walked)
