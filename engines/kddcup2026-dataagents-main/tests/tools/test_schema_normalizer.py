"""Unit tests for `_normalize_for_vllm`.

Three pinned transforms (see `registry.py`):
1. Strip every `title` key at every depth.
2. Collapse `{"anyOf": [{"type": T}, {"type": "null"}]}` (both orderings) into
   `{"type": [T, "null"]}`, preserving sibling keys.
3. Inject `required: []` on top-level object schemas that don't declare one.

Anything else passes through untouched.
"""

from __future__ import annotations

from agents.tools.schema_normalize import normalize_for_vllm as _normalize_for_vllm


def test_strip_title_at_top_level_and_in_properties() -> None:
    schema = {
        "title": "MyModel",
        "type": "object",
        "properties": {
            "path": {"title": "Path", "type": "string"},
            "limit": {"title": "Limit", "type": "integer"},
        },
        "required": ["path"],
    }
    normalized = _normalize_for_vllm(schema)
    assert "title" not in normalized
    assert "title" not in normalized["properties"]["path"]
    assert "title" not in normalized["properties"]["limit"]
    assert normalized["properties"]["path"] == {"type": "string"}
    assert normalized["required"] == ["path"]


def test_strip_title_inside_array_items() -> None:
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "title": "Row",
                    "type": "array",
                    "items": {"title": "Cell", "type": "string"},
                },
            }
        },
        "required": [],
    }
    normalized = _normalize_for_vllm(schema)
    item_schema = normalized["properties"]["rows"]["items"]
    assert "title" not in item_schema
    assert "title" not in item_schema["items"]


def test_anyof_nullable_collapsed_to_type_list() -> None:
    schema = {
        "type": "object",
        "properties": {
            "keyword": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "kw desc",
            }
        },
        "required": [],
    }
    normalized = _normalize_for_vllm(schema)
    keyword = normalized["properties"]["keyword"]
    assert keyword["type"] == ["string", "null"]
    assert "anyOf" not in keyword
    assert keyword["default"] is None
    assert keyword["description"] == "kw desc"


def test_anyof_nullable_collapsed_when_null_listed_first() -> None:
    schema = {
        "type": "object",
        "properties": {
            "keyword": {
                "anyOf": [{"type": "null"}, {"type": "string"}],
                "default": None,
            }
        },
        "required": [],
    }
    normalized = _normalize_for_vllm(schema)
    assert normalized["properties"]["keyword"]["type"] == ["string", "null"]


def test_anyof_with_more_than_two_branches_is_not_collapsed() -> None:
    """Heterogeneous cell schema in `answer.rows` keeps the `anyOf` shape."""
    cell_schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
        ]
    }
    schema = {
        "type": "object",
        "properties": {"cell": cell_schema},
        "required": [],
    }
    normalized = _normalize_for_vllm(schema)
    assert normalized["properties"]["cell"] == cell_schema


def test_anyof_with_complex_branch_is_not_collapsed() -> None:
    """Branches carrying anything beyond `type` are out of scope (keep anyOf)."""
    schema = {
        "type": "object",
        "properties": {
            "field": {
                "anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}],
            }
        },
        "required": [],
    }
    normalized = _normalize_for_vllm(schema)
    assert "anyOf" in normalized["properties"]["field"]
    assert "type" not in normalized["properties"]["field"]


def test_required_empty_injected_when_missing() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    normalized = _normalize_for_vllm(schema)
    assert normalized["required"] == []


def test_required_not_overwritten_when_present() -> None:
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    normalized = _normalize_for_vllm(schema)
    assert normalized["required"] == ["x"]


def test_minimum_and_default_siblings_preserved() -> None:
    schema = {
        "type": "object",
        "properties": {
            "max_rows": {
                "type": "integer",
                "minimum": 1,
                "default": 20,
                "description": "rows desc",
            }
        },
        "required": [],
    }
    normalized = _normalize_for_vllm(schema)
    assert normalized["properties"]["max_rows"] == {
        "type": "integer",
        "minimum": 1,
        "default": 20,
        "description": "rows desc",
    }


def test_unrelated_structures_pass_through_unchanged() -> None:
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "default": 200},
            "code": {"type": "string"},
        },
        "required": ["code"],
        "additionalProperties": False,
    }
    normalized = _normalize_for_vllm(schema)
    assert normalized == schema


def test_caller_input_is_not_mutated() -> None:
    schema: dict[str, object] = {
        "title": "X",
        "type": "object",
        "properties": {"a": {"title": "A", "type": "string"}},
    }
    snapshot = {
        "title": "X",
        "type": "object",
        "properties": {"a": {"title": "A", "type": "string"}},
    }
    _normalize_for_vllm(schema)
    assert schema == snapshot
