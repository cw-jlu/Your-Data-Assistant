"""run_explorer：video_findings 提取、video_tool 穿透。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.explorer.runner import _synthesize_findings, run_explorer
from agents.llm.types import ModelMessage, ModelResponse, ModelToolCall
from agents.tools.registry import FunctionTool, ToolExecutionResult
from tests.helpers.scripted_adapters import ScriptedNativeModelAdapter


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="What is X?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _make_scale_task(tmp_path: Path) -> PublicTask:
    question = "管理基金规模超过100亿的基金经理最高学历分布情况"
    task_dir = tmp_path / "task_3"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "knowledge.md").write_text(
        "### Use Case 3\n"
        f"Business Question: {question}\n"
        "Formula: `mf_fmscaleanalysisn.totalfundnv`\n",
        encoding="utf-8",
    )
    doc_dir = context_dir / "doc"
    doc_dir.mkdir()
    (doc_dir / "mf_fmscaleanalysisn.pdf").write_bytes(b"%PDF-1.4\n")
    return PublicTask(
        record=TaskRecord(task_id="task_3", difficulty="easy", question=question),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _call(call_id: str, name: str, arguments: dict) -> ModelResponse:
    raw = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return ModelResponse(
        content="",
        tool_calls=[ModelToolCall(id=call_id, name=name, arguments=arguments)],
        raw_response="",
        raw_tool_calls=[raw],
    )


def _stub_video_tool() -> FunctionTool:
    # json_schema 非 None：to_openai_tools 会跳过无 schema 的工具，
    # 同源断言需要 explore_video 出现在每轮请求的广告里
    return FunctionTool(
        name="explore_video",
        description="stub",
        json_schema={"type": "object", "properties": {}},
        handler=lambda _t, _a: ToolExecutionResult(
            ok=True,
            content={"video_status": "ok", "findings": {"summary": "chart video"}},
        ),
    )


class _RecordingAdapter:
    def __init__(self, inner: ScriptedNativeModelAdapter) -> None:
        self.inner = inner
        self.seen: list[list[ModelMessage]] = []
        self.tools_seen: list[Any] = []

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None, **kwargs: Any
    ) -> ModelResponse:
        self.seen.append(messages)
        self.tools_seen.append(tools)
        return self.inner.complete(messages, tools=tools, **kwargs)


def _advertised_names_per_turn(adapter: _RecordingAdapter) -> list[set[str]]:
    """渲染每轮下推 registry 的 OpenAI tools 广告，取工具名集合。"""
    per_turn: list[set[str]] = []
    for tools in adapter.tools_seen:
        assert tools is not None, "agent must push its registry on every turn"
        per_turn.append({entry["function"]["name"] for entry in tools.to_openai_tools()})
    return per_turn


def test_video_findings_extracted_from_steps(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    adapter = ScriptedNativeModelAdapter(
        [
            _call("v1", "explore_video", {"path": "demo.mp4", "instructions": "look"}),
            _call("r1", "report", {"knowledge": {"from_video": "chart video"}}),
        ]
    )
    result = run_explorer(task, task.question, adapter, video_tool=_stub_video_tool())
    assert result.success is True
    assert result.video_findings is not None
    assert any("chart video" in json.dumps(v) for v in result.video_findings)


def test_no_video_tool_no_video_findings(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    adapter = ScriptedNativeModelAdapter([_call("r1", "report", {"knowledge": {}})])
    result = run_explorer(task, task.question, adapter)
    assert result.success is True
    assert result.video_findings is None


def test_each_turn_advertises_video_tool_when_injected(tmp_path: Path) -> None:
    """同源行为：注入 video_tool 的 run，每轮请求的工具广告都含 explore_video。"""
    task = _make_task(tmp_path)
    adapter = _RecordingAdapter(
        ScriptedNativeModelAdapter(
            [
                _call("v1", "explore_video", {"path": "demo.mp4", "instructions": "look"}),
                _call("r1", "report", {"knowledge": {"from_video": "chart video"}}),
            ]
        )
    )
    run_explorer(task, task.question, adapter, video_tool=_stub_video_tool())
    per_turn = _advertised_names_per_turn(adapter)
    assert len(per_turn) == 2
    assert all("explore_video" in names for names in per_turn)


def test_each_turn_omits_video_tool_when_not_injected(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    adapter = _RecordingAdapter(
        ScriptedNativeModelAdapter([_call("r1", "report", {"knowledge": {}})])
    )
    run_explorer(task, task.question, adapter)
    per_turn = _advertised_names_per_turn(adapter)
    assert per_turn
    assert all("explore_video" not in names for names in per_turn)


def test_report_rejection_forces_explorer_to_resubmit_etl_sources(tmp_path: Path) -> None:
    task = _make_scale_task(tmp_path)
    adapter = ScriptedNativeModelAdapter(
        [
            _call(
                "bad_report",
                "report",
                {
                    "files": [{"path": "json/mf_fmretscaleanalysis.json", "format": "JSON"}],
                    "schema_map": {"mf_fmretscaleanalysis": {"columns": {}}},
                    "knowledge": {
                        "knowledge_md_evidence": [
                            {
                                "source": "knowledge.md",
                                "kind": "formula",
                                "text": "`mf_fmscaleanalysisn.totalfundnv`",
                            }
                        ]
                    },
                    "etl_sources": [],
                },
            ),
            _call(
                "good_report",
                "report",
                {
                    "files": [{"path": "json/mf_fmretscaleanalysis.json", "format": "JSON"}],
                    "schema_map": {"mf_fmretscaleanalysis": {"columns": {}}},
                    "knowledge": {
                        "knowledge_md_evidence": [
                            {
                                "source": "knowledge.md",
                                "kind": "formula",
                                "text": "`mf_fmscaleanalysisn.totalfundnv`",
                            }
                        ]
                    },
                    "etl_sources": [
                        {
                            "path": "doc/mf_fmscaleanalysisn.pdf",
                            "reason": (
                                "knowledge.md exact table mf_fmscaleanalysisn has no "
                                "structured source and exists as a same-stem PDF"
                            ),
                        }
                    ],
                },
            ),
        ]
    )

    result = run_explorer(task, task.question, adapter)

    assert result.success is True
    findings = json.loads(result.findings)
    assert findings["etl_sources"] == [
        {
            "path": "doc/mf_fmscaleanalysisn.pdf",
            "reason": (
                "knowledge.md exact table mf_fmscaleanalysisn has no "
                "structured source and exists as a same-stem PDF"
            ),
        }
    ]


def test_final_budget_prompt_names_report_for_explorer(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    filler = [_call(f"f{i}", "inspect_files", {}) for i in range(9)]
    adapter = _RecordingAdapter(
        ScriptedNativeModelAdapter(
            [
                *filler,
                _call("r1", "report", {"knowledge": {"done": True}}),
            ]
        )
    )

    result = run_explorer(task, task.question, adapter, max_steps=10)

    assert result.success is True
    critical_messages = [
        m.content
        for m in adapter.seen[9]
        if isinstance(m.content, str) and "CRITICAL:" in m.content
    ]
    assert len(critical_messages) == 1
    assert "call `report` NOW" in critical_messages[0]
    assert "`answer`" not in critical_messages[0]


def test_explorer_stops_after_final_report_retry_blocks(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    filler = [_call(f"f{i}", "inspect_files", {}) for i in range(9)]
    adapter = _RecordingAdapter(
        ScriptedNativeModelAdapter(
            [
                *filler,
                _call("bad1", "inspect_files", {}),
                _call("bad2", "inspect_files", {}),
            ]
        )
    )

    result = run_explorer(task, task.question, adapter, max_steps=10)

    assert result.success is True
    assert result.fallback_used is True
    assert "files" in json.loads(result.findings)
    assert len(adapter.seen) == 11


def test_synthesize_findings_keeps_json_and_text_preview_file_observations() -> None:
    synthesized = _synthesize_findings(
        [
            SimpleNamespace(
                action="preview_file",
                action_input={"path": "json/Patient.json"},
                ok=True,
                observation={
                    "content": {
                        "kind": "object",
                        "key_count": 2,
                        "keys": ["patients", "meta"],
                        "value_preview": {"patients": {"kind": "array", "length": 3}},
                        "truncated": False,
                    }
                },
            ),
            SimpleNamespace(
                action="preview_file",
                action_input={"path": "notes.md"},
                ok=True,
                observation={
                    "content": {
                        "format": "text",
                        "total_lines": 12,
                        "head": ["# Knowledge", "A = Alpha"],
                        "has_more": False,
                    }
                },
            ),
            SimpleNamespace(
                action="preview_file",
                action_input={"path": "knowledge.md"},
                ok=True,
                observation={
                    "content": {
                        "format": "text",
                        "total_lines": 3,
                        "lines": ["# Knowledge", "A = Alpha", "B = Beta"],
                    }
                },
            ),
        ]
    )

    findings = json.loads(synthesized)
    assert findings["json_previews"]["json/Patient.json"] == {
        "kind": "object",
        "key_count": 2,
        "keys": ["patients", "meta"],
        "value_preview": {"patients": {"kind": "array", "length": 3}},
        "truncated": False,
    }
    assert findings["text_previews"]["notes.md"] == {
        "total_lines": 12,
        "head": ["# Knowledge", "A = Alpha"],
        "has_more": False,
    }
    assert findings["text_previews"]["knowledge.md"] == {
        "total_lines": 3,
        "lines": ["# Knowledge", "A = Alpha", "B = Beta"],
    }
