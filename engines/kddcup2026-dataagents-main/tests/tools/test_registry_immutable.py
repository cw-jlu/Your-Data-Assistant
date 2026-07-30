"""测试 ToolRegistry 在构造完成后不可变。

覆盖点：
- 默认工具集按规范的注册顺序排布
- 仅 `answer` 的 FunctionTool.is_terminal 为 True
- registry.definitions 是 MappingProxyType，对其增/删/改均抛 TypeError
- registry 自身是 frozen dataclass，重新赋值 definitions 字段也会被拒绝
- 单个 FunctionTool 同样 frozen，无法在构造后修改字段
- 即使外部传入普通 dict 后再去修改源 dict，也无法绕过只读视图（防御性浅拷贝）
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from agents.tools.registry import (
    FunctionTool,
    ToolExecutionResult,
    ToolRegistry,
    create_default_tool_registry,
)

_EXPECTED_ORDER = [
    "inspect_files",
    "run_etl",
    "preview_file",
    "execute_context_sql",
    "grep_context",
    "execute_python",
    "answer",
]


def _noop_handler(_task, _action_input):
    return ToolExecutionResult(ok=True, content={})


def _sample_definition(name: str = "extra") -> FunctionTool:
    return FunctionTool(
        name=name,
        description="extra",
        json_schema=None,
        handler=_noop_handler,
    )


def test_default_registry_preserves_registration_order() -> None:
    registry = create_default_tool_registry()
    assert list(registry.definitions.keys()) == _EXPECTED_ORDER


def test_only_answer_is_terminal() -> None:
    registry = create_default_tool_registry()
    terminal_names = [
        definition.name for definition in registry.definitions.values() if definition.is_terminal
    ]
    assert terminal_names == ["answer"]


def test_definitions_is_mapping_proxy() -> None:
    registry = create_default_tool_registry()
    assert isinstance(registry.definitions, MappingProxyType)


def test_definitions_reject_item_assignment() -> None:
    registry = create_default_tool_registry()
    with pytest.raises(TypeError):
        registry.definitions["new_tool"] = _sample_definition("new_tool")  # type: ignore[index]


def test_definitions_reject_overwriting_existing_tool() -> None:
    registry = create_default_tool_registry()
    original = registry.definitions["answer"]
    with pytest.raises(TypeError):
        registry.definitions["answer"] = _sample_definition("answer")  # type: ignore[index]
    assert registry.definitions["answer"] is original


def test_definitions_reject_deletion() -> None:
    registry = create_default_tool_registry()
    with pytest.raises(TypeError):
        del registry.definitions["answer"]  # type: ignore[attr-defined]
    assert "answer" in registry.definitions


def test_registry_is_frozen_dataclass() -> None:
    registry = create_default_tool_registry()
    with pytest.raises(FrozenInstanceError):
        registry.definitions = MappingProxyType({})  # type: ignore[misc]


def test_tool_definition_is_frozen() -> None:
    definition = _sample_definition()
    with pytest.raises(FrozenInstanceError):
        definition.name = "renamed"  # type: ignore[misc]


def test_external_dict_mutations_do_not_leak_into_registry() -> None:
    # 外部构造时持有源 dict 引用；构造完成后修改源 dict 不应影响 registry。
    source: dict[str, FunctionTool] = {"alpha": _sample_definition("alpha")}
    registry = ToolRegistry(definitions=source)

    source["beta"] = _sample_definition("beta")
    source.pop("alpha", None)

    assert list(registry.definitions.keys()) == ["alpha"]
    assert "beta" not in registry.definitions


def test_external_proxy_wrapping_mutable_dict_does_not_leak() -> None:
    # 调用方先把可变 dict 包成 MappingProxyType 再传入：
    # registry 必须对其做防御性浅拷贝，否则源 dict 的后续修改会透过 proxy 漏进来。
    source: dict[str, FunctionTool] = {"alpha": _sample_definition("alpha")}
    proxy = MappingProxyType(source)
    registry = ToolRegistry(definitions=proxy)

    source["beta"] = _sample_definition("beta")
    source.pop("alpha", None)

    assert list(registry.definitions.keys()) == ["alpha"]
    assert "beta" not in registry.definitions
