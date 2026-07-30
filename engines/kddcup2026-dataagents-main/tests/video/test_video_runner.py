"""Video runner: scripted report extraction and failure handling."""

from __future__ import annotations

import json
from pathlib import Path

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.llm.types import ModelResponse, ModelToolCall
from agents.video.prompt import VIDEO_AGENT_SYSTEM_PROMPT, build_video_task_text
from agents.video.runner import run_video_agent
from tests.helpers.scripted_adapters import ScriptedNativeModelAdapter


def _make_task(tmp_path: Path) -> tuple[PublicTask, Path]:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    video = context_dir / "demo.mp4"
    video.write_bytes(b"\x00\x01\x02")
    task = PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="What is Q3 revenue?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )
    return task, video


def _report_response(arguments: dict) -> ModelResponse:
    raw = {
        "id": "r1",
        "type": "function",
        "function": {"name": "report", "arguments": json.dumps(arguments)},
    }
    return ModelResponse(
        content="watched",
        tool_calls=[ModelToolCall(id="r1", name="report", arguments=arguments)],
        raw_response="watched",
        raw_tool_calls=[raw],
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


def test_successful_report_extracted_despite_failure_reason_trap(tmp_path: Path) -> None:
    task, video = _make_task(tmp_path)
    adapter = ScriptedNativeModelAdapter(
        [
            _report_response(
                {
                    "summary": "a chart video",
                    "extracted_data": {
                        "rules": {
                            "canonical": {
                                "filters": [{"field": "quarter", "operator": "=", "value": "Q3"}]
                            }
                        },
                        "displayed_samples": {"kpi": 42},
                    },
                }
            )
        ]
    )
    result = run_video_agent(task, video, "transcribe the KPI", adapter)
    assert result.success is True
    findings = json.loads(result.findings)
    assert findings["summary"] == "a chart video"
    assert findings["extracted_data"]["rules"]["canonical"]["filters"] == [
        {"field": "quarter", "operator": "=", "value": "Q3"}
    ]
    assert findings["extracted_data"]["displayed_samples"] == {"kpi": 42}


def test_report_without_extracted_data_gets_empty_contract(tmp_path: Path) -> None:
    task, video = _make_task(tmp_path)
    adapter = ScriptedNativeModelAdapter([_report_response({"summary": "a chart video"})])

    result = run_video_agent(task, video, "transcribe the KPI", adapter)

    assert result.success is True
    findings = json.loads(result.findings)
    assert findings["extracted_data"] == {"rules": _empty_rules(), "displayed_samples": {}}


def test_no_report_returns_failure(tmp_path: Path) -> None:
    task, video = _make_task(tmp_path)
    text_only = ModelResponse(content="just prose", tool_calls=[], raw_response="just prose")
    adapter = ScriptedNativeModelAdapter([text_only, text_only])
    result = run_video_agent(task, video, "transcribe", adapter)
    assert result.success is False
    assert result.failure_reason


def test_task_text_delimits_brief_and_restates_contract() -> None:
    text = build_video_task_text("Q?", "look at the bar chart")
    assert "<analysis_brief>\nlook at the bar chart\n</analysis_brief>" in text
    assert "Q?" in text
    assert "report" in text
    assert "<analysis_brief>" in text and text.index("Q?") < text.index("<analysis_brief>")


def test_system_prompt_is_static_and_declares_brief_as_data() -> None:
    assert "{" not in VIDEO_AGENT_SYSTEM_PROMPT
    assert "analysis_brief" in VIDEO_AGENT_SYSTEM_PROMPT
    assert "report" in VIDEO_AGENT_SYSTEM_PROMPT
