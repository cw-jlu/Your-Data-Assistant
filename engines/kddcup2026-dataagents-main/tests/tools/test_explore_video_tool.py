"""explore_video 工具：路径/类型/大小三类错误 + happy path + 子代理失败传播。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import agents.runtime.media as media_mod
import agents.video.runner as video_runner_mod
from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools.explore_video import ExploreVideoInput, create_explore_video_tool_definition
from agents.video.runner import VideoAgentResult


class _StubAdapter:
    def complete(
        self, messages, *, tools=None, **kwargs
    ):  # pragma: no cover - never called in these tests
        raise AssertionError("video model must not be called")


@pytest.fixture
def task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "demo.mp4").write_bytes(b"\x00" * 100)
    (context_dir / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="q"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _args(path: str) -> dict:
    return {"path": path, "instructions": "transcribe everything"}


def test_happy_path_returns_parsed_findings(
    task: PublicTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    def fake_run(t, video_path, instructions, model, **kwargs):
        seen.update(path=video_path, instructions=instructions)
        return VideoAgentResult(
            findings=json.dumps(
                {
                    "summary": "s",
                    "extracted_data": {
                        "rules": {},
                        "displayed_samples": {"k": 1},
                    },
                }
            ),
            steps_used=1,
            success=True,
        )

    monkeypatch.setattr(video_runner_mod, "run_video_agent", fake_run)
    tool = create_explore_video_tool_definition(_StubAdapter())
    result = tool.handler(task, ExploreVideoInput(**_args("demo.mp4")))
    assert result.ok is True
    assert result.content["findings"]["summary"] == "s"
    assert seen["instructions"] == "transcribe everything"
    assert seen["path"].name == "demo.mp4"


def test_filter_recipe_attached_when_rules_have_filter_shape(
    task: PublicTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rules with field+operator+threshold trigger a recipe attached to the observation."""

    def fake_run(t, video_path, instructions, model, **kwargs):
        return VideoAgentResult(
            findings=json.dumps(
                {
                    "summary": "dashboard config",
                    "extracted_data": {
                        "rules": {
                            "field": "dailybenchgr",
                            "operator": ">",
                            "threshold": "0.00%",
                            "sort_field": "dailybenchgr",
                            "sort_direction": "descending",
                            "export_field": "SecuAbbr",
                            "table": "mf_benchmarkgrowthrate",
                        },
                        "displayed_samples": {},
                    },
                }
            ),
            steps_used=2,
            success=True,
        )

    monkeypatch.setattr(video_runner_mod, "run_video_agent", fake_run)
    tool = create_explore_video_tool_definition(_StubAdapter())
    result = tool.handler(task, ExploreVideoInput(**_args("demo.mp4")))
    assert result.ok is True
    recipe = result.content.get("filter_recipe")
    assert recipe is not None
    assert "dailybenchgr" in recipe
    assert "VERIFY" in recipe
    assert "ascending=False" in recipe


def test_filter_recipe_absent_when_rules_too_sparse(
    task: PublicTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rules with only metadata (no field/op/sort/structural anchor) → no recipe key."""

    def fake_run(t, video_path, instructions, model, **kwargs):
        return VideoAgentResult(
            findings=json.dumps(
                {
                    "summary": "narration only",
                    "extracted_data": {
                        "rules": {"task_name": "Configure something", "status": "active"},
                        "displayed_samples": {},
                    },
                }
            ),
            steps_used=1,
            success=True,
        )

    monkeypatch.setattr(video_runner_mod, "run_video_agent", fake_run)
    tool = create_explore_video_tool_definition(_StubAdapter())
    result = tool.handler(task, ExploreVideoInput(**_args("demo.mp4")))
    assert result.ok is True
    assert "filter_recipe" not in result.content


def test_missing_path_returns_error(task: PublicTask) -> None:
    tool = create_explore_video_tool_definition(_StubAdapter())
    result = tool.handler(task, ExploreVideoInput(**_args("nope.mp4")))
    assert result.ok is False and "error" in result.content


def test_non_video_extension_rejected(task: PublicTask) -> None:
    tool = create_explore_video_tool_definition(_StubAdapter())
    result = tool.handler(task, ExploreVideoInput(**_args("data.csv")))
    assert result.ok is False
    assert "not a video file" in result.content["error"]


def test_oversized_video_rejected(task: PublicTask, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_mod, "VIDEO_MAX_BYTES", 10)
    tool = create_explore_video_tool_definition(_StubAdapter())
    result = tool.handler(task, ExploreVideoInput(**_args("demo.mp4")))
    assert result.ok is False
    assert "exceeds" in result.content["error"]


def test_subagent_failure_propagates(task: PublicTask, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        video_runner_mod,
        "run_video_agent",
        lambda *a, **k: VideoAgentResult(
            findings="", steps_used=3, success=False, failure_reason="no report"
        ),
    )
    tool = create_explore_video_tool_definition(_StubAdapter())
    result = tool.handler(task, ExploreVideoInput(**_args("demo.mp4")))
    assert result.ok is False
    assert result.content["video_status"] == "failed"
