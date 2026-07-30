from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from data_agent_baseline.agents.ambiguity_analyzer import (
    AMBIGUITY_TYPES,
    analyze_ambiguity,
    _parse_ambiguity_response,
    _validate_ambiguities,
    _validate_non_ambiguous_candidates,
)


# ---------------------------------------------------------------------------
# Constant tests
# ---------------------------------------------------------------------------


def test_ambiguity_types_enum() -> None:
    assert "field_binding" in AMBIGUITY_TYPES
    assert "metric_definition" in AMBIGUITY_TYPES
    assert "entity_resolution" in AMBIGUITY_TYPES
    assert "filter_semantics" in AMBIGUITY_TYPES
    assert "time_range" in AMBIGUITY_TYPES
    assert "grain" in AMBIGUITY_TYPES
    assert "join_path" in AMBIGUITY_TYPES
    assert "output_format" in AMBIGUITY_TYPES
    assert len(AMBIGUITY_TYPES) == 12


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


def test_parse_ambiguity_response_full() -> None:
    parsed = _parse_ambiguity_response(
        json.dumps({
            "question_intent": {
                "entities": ["Alex Yoong"],
                "filters": ["track < 20"],
                "metrics": [],
                "requested_output": "race name",
                "grain": "per race",
            },
            "ambiguities": [
                {
                    "id": "amb_001",
                    "phrase": "track number",
                    "type": "field_binding",
                    "clarifying_question": "Does track number mean round?",
                    "required_verification": ["probe distinct values of races.csv.round"],
                }
            ],
            "resolved_by_knowledge": [
                {"phrase": "score", "chosen_meaning": "official score", "evidence": "doc defines score"},
            ],
            "non_ambiguous_candidates": [
                {"phrase": "forename", "candidate_fields": ["drivers.json.forename"], "note": "clear"},
            ],
        })
    )
    assert parsed is not None
    assert parsed["question_intent"]["entities"] == ["Alex Yoong"]
    assert parsed["question_intent"]["grain"] == "per race"
    assert len(parsed["ambiguities"]) == 1
    assert parsed["ambiguities"][0]["id"] == "amb_001"
    assert parsed["ambiguities"][0]["type"] == "field_binding"
    assert len(parsed["resolved_by_knowledge"]) == 1
    assert len(parsed["non_ambiguous_candidates"]) == 1


def test_parse_ambiguity_response_minimal() -> None:
    parsed = _parse_ambiguity_response(
        json.dumps({
            "question_intent": {"entities": [], "filters": [], "metrics": [],
                                "requested_output": "count", "grain": "unknown"},
            "ambiguities": [],
            "resolved_by_knowledge": [],
            "non_ambiguous_candidates": [],
        })
    )
    assert parsed is not None
    assert parsed["question_intent"]["entities"] == []
    assert parsed["ambiguities"] == []


def test_parse_ambiguity_response_non_json() -> None:
    assert _parse_ambiguity_response("hello world") is None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_validate_ambiguities_strips_invalid_types() -> None:
    ambiguities = [
        {"id": "amb_001", "phrase": "x", "type": "field_binding"},
        {"id": "amb_002", "phrase": "y", "type": "not_a_real_type"},
        {"id": "amb_003", "phrase": "z", "type": "metric_definition"},
    ]
    result = _validate_ambiguities(ambiguities)
    assert len(result) == 2
    types = {a["type"] for a in result}
    assert types == {"field_binding", "metric_definition"}


def test_validate_ambiguities_strips_missing_id() -> None:
    ambiguities = [
        {"phrase": "x", "type": "field_binding"},
        {"id": "", "phrase": "y", "type": "metric_definition"},
    ]
    result = _validate_ambiguities(ambiguities)
    assert result == []


def test_validate_non_ambiguous_candidates_strips_invalid_fields() -> None:
    entries = [
        {"phrase": "valid", "candidate_fields": ["a.csv.col_0"]},
        {"phrase": "mixed", "candidate_fields": ["a.csv.col_0", "x.csv.no_such"]},
    ]
    schemas = [
        {"asset_path": "a.csv", "kind": "csv", "fields": [{"name": "col_0", "type": "integer"}]},
    ]
    result = _validate_non_ambiguous_candidates(entries, schemas)
    assert len(result) == 2
    assert result[0]["candidate_fields"] == ["a.csv.col_0"]
    assert result[1]["candidate_fields"] == ["a.csv.col_0"]


def test_validate_non_ambiguous_candidates_all_invalid_returns_empty() -> None:
    entries = [
        {"phrase": "bad", "candidate_fields": ["x.csv.no_such"]},
    ]
    schemas = [
        {"asset_path": "real.csv", "kind": "csv", "fields": [{"name": "field", "type": "text"}]},
    ]
    result = _validate_non_ambiguous_candidates(entries, schemas)
    assert result == []


# ---------------------------------------------------------------------------
# analyze_ambiguity tests
# ---------------------------------------------------------------------------


def test_analyze_ambiguity_without_schemas(monkeypatch) -> None:
    def fake_invoke(model, messages):
        return AIMessage(content=json.dumps({
            "question_intent": {"entities": ["x"], "filters": [], "metrics": [],
                                "requested_output": "y", "grain": "per record"},
            "ambiguities": [],
            "resolved_by_knowledge": [],
            "non_ambiguous_candidates": [],
        }))

    monkeypatch.setattr(
        "data_agent_baseline.agents.ambiguity_analyzer.invoke_model_with_retries",
        fake_invoke,
    )
    result = analyze_ambiguity(model=MagicMock(), question="test?")
    assert result["question_intent"]["entities"] == ["x"]
    assert result["ambiguities"] == []
    assert result["non_ambiguous_candidates"] == []


def test_analyze_ambiguity_with_schemas(monkeypatch) -> None:
    def fake_invoke(model, messages):
        return AIMessage(content=json.dumps({
            "question_intent": {"entities": ["Alex Yoong"], "filters": [],
                                "metrics": [], "requested_output": "race name", "grain": "per race"},
            "ambiguities": [
                {
                    "id": "amb_001",
                    "phrase": "Alex Yoong",
                    "type": "entity_resolution",
                    "clarifying_question": "Is Alex Yoong the forename or full name?",
                    "required_verification": ["probe distinct values of forename and surname"],
                }
            ],
            "resolved_by_knowledge": [],
            "non_ambiguous_candidates": [
                {"phrase": "forename", "candidate_fields": ["data/drivers.json.forename"], "note": "clear"},
            ],
        }))

    monkeypatch.setattr(
        "data_agent_baseline.agents.ambiguity_analyzer.invoke_model_with_retries",
        fake_invoke,
    )
    schemas = [
        {"asset_path": "data/drivers.json", "kind": "json",
         "fields": [{"name": "forename", "type": "text"}, {"name": "surname", "type": "text"}]},
    ]
    result = analyze_ambiguity(model=MagicMock(), question="Which race was Alex Yoong in?", schemas=schemas)
    assert result["question_intent"]["entities"] == ["Alex Yoong"]
    assert len(result["ambiguities"]) == 1
    assert result["ambiguities"][0]["type"] == "entity_resolution"
    assert len(result["non_ambiguous_candidates"]) == 1


def test_analyze_ambiguity_fallback(monkeypatch) -> None:
    def fake_invoke(model, messages):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(
        "data_agent_baseline.agents.ambiguity_analyzer.invoke_model_with_retries",
        fake_invoke,
    )
    result = analyze_ambiguity(model=MagicMock(), question="test?")
    assert result["question_intent"]["entities"] == []
    assert result["ambiguities"] == []
    assert result["resolved_by_knowledge"] == []
    assert result["non_ambiguous_candidates"] == []
