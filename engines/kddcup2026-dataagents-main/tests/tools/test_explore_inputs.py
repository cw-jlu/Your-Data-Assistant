"""ExploreInput 与 ExploreVideoInput 的契约。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools.explore import ExploreInput, _ensure_column_conversions
from agents.tools.explore_video import EXPLORE_VIDEO_SCHEMA, ExploreVideoInput
from agents.tools.units import load_conversions


def test_explore_input_takes_no_params() -> None:
    ExploreInput.model_validate({})
    with pytest.raises(ValidationError):
        ExploreInput.model_validate({"unknown": 1})


def test_explore_video_input_requires_path_and_instructions() -> None:
    ok = ExploreVideoInput.model_validate(
        {"path": "demo.mp4", "instructions": "transcribe the bar chart"}
    )
    assert ok.path == "demo.mp4"
    with pytest.raises(ValidationError):
        ExploreVideoInput.model_validate({"path": "demo.mp4"})
    with pytest.raises(ValidationError):
        ExploreVideoInput.model_validate({"instructions": "x"})


def test_explore_video_schema_forbids_extras() -> None:
    assert EXPLORE_VIDEO_SCHEMA["additionalProperties"] is False
    assert set(EXPLORE_VIDEO_SCHEMA["required"]) == {"path", "instructions"}


def test_load_conversions_matches_target_factor_case_insensitively(tmp_path) -> None:
    units_path = tmp_path / "sample_units.json"
    units_path.write_text(
        json.dumps(
            {
                "Amount": "元",
                "_target_amount": "万元",
                "_factor_AMOUNT": "0.0001",
            }
        ),
        encoding="utf-8",
    )

    conversions = load_conversions(units_path)

    assert [(c.field, c.source, c.target, c.factor) for c in conversions] == [
        ("Amount", "元", "万元", 0.0001)
    ]


def test_explore_column_conversions_match_schema_columns_case_insensitively(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    monkeypatch.setattr("agents.config.ETL_SCRATCH_ROOT", scratch)
    task_dir = tmp_path / "task_32"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    task = PublicTask(
        record=TaskRecord(task_id="task_32", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    cache_dir = scratch / task.task_id / "_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "qt_dailyquote_units.json").write_text(
        json.dumps(
            {
                "secucode": "元",
                "_target_SecuCode": "万元",
                "_factor_SecuCode": "0.0001",
            }
        ),
        encoding="utf-8",
    )
    content = {
        "schema_map": {
            "context/csv/qt_dailyquote.csv": {
                "columns": {
                    "SecuCode": {"description": "Security code."},
                }
            }
        }
    }

    _ensure_column_conversions(task, content)

    entry = content["schema_map"]["context/csv/qt_dailyquote.csv"]
    assert entry["column_conversions"]["secucode"] == "CONVERT: ×0.0001 (元 → 万元)"
    assert "[UNIT: stored as 元, submit as 万元" in entry["columns"]["SecuCode"]["description"]
