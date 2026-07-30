"""Video agent registry: report-only terminal tool with structured rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.video.registry import (
    VIDEO_REPORT_MAX_BYTES,
    VideoReportInput,
    create_video_tool_registry,
)


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="q"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _empty_canonical_rules() -> dict:
    return {
        "filters": [],
        "metrics": [],
        "group_by": [],
        "sort_by": [],
        "periods": [],
        "identifiers": {},
        "tables": [],
        "output_fields": [],
        "snapshot_dates": [],
    }


def _empty_rules() -> dict:
    return {
        "canonical": _empty_canonical_rules(),
        "rule_items": [],
        "raw_observations": [],
        "task_specific": {},
    }


def test_registry_contains_only_terminal_report() -> None:
    registry = create_video_tool_registry()
    assert list(registry.definitions) == ["report"]
    assert registry.definitions["report"].is_terminal is True
    assert registry.definitions["report"].input_model is not None


def test_valid_report_roundtrips_structured_rules(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {
            "summary": "dashboard recording",
            "timeline": [{"time": "00:42", "event": "bar chart"}],
            "extracted_data": {
                "rules": {
                    "canonical": {
                        "filters": [{"field": "quarter", "op": "=", "value": "Q3"}],
                        "logic": "AND",
                        "group_by": "region",
                        "identifiers": {"batch_id": "B-1"},
                    },
                    "rule_items": [
                        {
                            "id": "r1",
                            "kind": "filter",
                            "description": "quarter equals Q3",
                            "fields": "quarter",
                        }
                    ],
                    "raw_observations": "Rule text: quarter = Q3",
                },
                "displayed_samples": {"q3_revenue": {"A": 10, "B": 20}},
            },
        },
    )
    assert result.ok is True and result.is_terminal is True
    findings = json.loads(result.content["findings"])
    rules = findings["extracted_data"]["rules"]
    assert findings["summary"] == "dashboard recording"
    assert rules["canonical"]["filters"] == [{"field": "quarter", "operator": "=", "value": "Q3"}]
    assert rules["canonical"]["logic"] == "AND"
    assert rules["canonical"]["group_by"] == ["region"]
    assert rules["canonical"]["identifiers"] == {"batch_id": "B-1"}
    assert rules["rule_items"][0]["fields"] == ["quarter"]
    assert rules["raw_observations"] == ["Rule text: quarter = Q3"]
    assert findings["extracted_data"]["displayed_samples"] == {"q3_revenue": {"A": 10, "B": 20}}
    assert findings["coverage"] == [] and findings["warnings"] == []


def test_legacy_flat_rules_are_preserved_in_task_specific(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {
            "summary": "dashboard recording",
            "extracted_data": {
                "rules": {"scheme_id": "Q3-FM-07", "threshold": "100.00"},
                "displayed_samples": {},
            },
        },
    )

    assert result.ok is True
    findings = json.loads(result.content["findings"])
    rules = findings["extracted_data"]["rules"]
    assert rules["canonical"] == _empty_canonical_rules()
    assert rules["task_specific"] == {"scheme_id": "Q3-FM-07", "threshold": "100.00"}


def test_displayed_samples_accepts_list_shape(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    samples = [{"rank": 1, "name": "Alpha"}, {"rank": 2, "name": "Beta"}]
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {
            "summary": "ranked panel",
            "extracted_data": {
                "rules": {"canonical": {"top_n": 2}},
                "displayed_samples": samples,
            },
        },
    )

    assert result.ok is True and result.is_terminal is True
    findings = json.loads(result.content["findings"])
    assert findings["extracted_data"]["displayed_samples"] == samples


def test_displayed_samples_schema_description_allows_arrays() -> None:
    schema = VideoReportInput.model_json_schema()
    description = schema["properties"]["extracted_data"]["description"]

    assert "structured object OR array" in description
    assert "MUST be a structured object. FORBIDDEN" not in description


def test_missing_extracted_data_defaults_to_empty_contract(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    result = registry.execute(_make_task(tmp_path), "report", {"summary": "dashboard"})

    assert result.ok is True
    findings = json.loads(result.content["findings"])
    assert findings["extracted_data"] == {"rules": _empty_rules(), "displayed_samples": {}}


def test_null_warnings_normalized_to_empty_list(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {
            "summary": "dashboard",
            "coverage": None,
            "warnings": None,
        },
    )

    assert result.ok is True
    findings = json.loads(result.content["findings"])
    assert findings["coverage"] == []
    assert findings["warnings"] == []


def test_stringified_extracted_data_and_single_coverage_normalized(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {
            "summary": "dashboard",
            "extracted_data": json.dumps(
                {
                    "rules": {"scheme_id": "Q3-FM-07"},
                    "displayed_samples": {"threshold": "100.00"},
                },
                ensure_ascii=False,
            ),
            "coverage": "Threshold details were visible but partly abbreviated.",
            "warnings": None,
        },
    )

    assert result.ok is True
    findings = json.loads(result.content["findings"])
    assert findings["extracted_data"]["rules"]["task_specific"] == {"scheme_id": "Q3-FM-07"}
    assert findings["extracted_data"]["displayed_samples"] == {"threshold": "100.00"}
    assert findings["coverage"] == ["Threshold details were visible but partly abbreviated."]
    assert findings["warnings"] == []


def test_flat_extracted_data_rejected(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    with pytest.raises(
        ValueError,
        match=r"report: extra field 'extracted_data\.kpi' is not permitted",
    ):
        registry.execute(
            _make_task(tmp_path),
            "report",
            {"summary": "dashboard", "extracted_data": {"kpi": 42}},
        )


def test_missing_summary_raises_single_line_error(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    with pytest.raises(ValueError, match="report: field 'summary' is required"):
        registry.execute(_make_task(tmp_path), "report", {})


def test_oversized_report_rejected_not_terminal(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    big = "x" * (VIDEO_REPORT_MAX_BYTES + 1)
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {"summary": "s", "extracted_data": {"rules": {"blob": big}}},
    )
    assert result.ok is False
    assert "exceeds" in result.content["error"]
    assert result.is_terminal is False
