from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from data_agent_baseline.agents.process_validator import (
    _build_process_validation_request,
    _parse_process_validator_response,
    validate_process,
)
from data_agent_baseline.agents.langgraph_runtime import _build_supporting_source_evidence


def test_process_validator_request_includes_semantic_inputs() -> None:
    request = _build_process_validation_request(
        question="What is the average?",
        answer={"columns": ["avg"], "rows": [[1.0]]},
        submission_context={"source_tool": "execute_probe_query", "source_tool_args": {"queries": ["SELECT 1"]}},
        supporting_source_evidence={"evidence_items": [{"tool": "read_doc"}]},
        submission_risk_report={"detected": [{"kind": "row_limit"}]},
        ambiguity_analysis={"ambiguities": [{"id": "amb_001"}]},
        recent_steps=[{"step_index": 1, "node": "tool"}],
        semantic_ledger={"intent_summary": "average"},
    )

    assert "Supporting Source Evidence" in request
    assert "Programmatic Submission Risk Report" not in request
    assert "Submission Source" in request
    assert "amb_001" in request
    assert "first-five-row answer preview" in request
    assert "Never treat a truncated preview as evidence" in request
    assert "audit reproducibility" in request


def test_process_validator_request_exposes_scalar_zero_preview() -> None:
    request = _build_process_validation_request(
        question="How many records qualify?",
        answer={"columns": ["count"], "rows": [[0]]},
    )

    assert '"values_preview_available": true' in request
    assert '"values_preview_truncated": false' in request
    assert '"rows_preview": [\n    [\n      0\n    ]\n  ]' in request
    assert '"scalar_value": 0' in request


def test_process_validator_request_exposes_first_five_answer_rows() -> None:
    request = _build_process_validation_request(
        question="List records.",
        answer={
            "columns": ["a", "b", "c", "d"],
            "rows": [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24],
            ],
        },
    )

    assert '"row_count": 6' in request
    assert '"column_count": 4' in request
    assert '"values_preview_available": true' in request
    assert '"values_preview_truncated": true' in request
    assert '"rows_preview":' in request
    assert "[\n      17,\n      18,\n      19,\n      20\n    ]" in request
    assert "[\n      21,\n      22,\n      23,\n      24\n    ]" not in request
    assert '"scalar_value":' not in request


def test_process_request_exposes_strict_v3_video_policy() -> None:
    request = _build_process_validation_request(
        question="What video rule applies?",
        strict_video_evidence=True,
    )

    assert "strict_v3_video_evidence: true" in request
    assert "visual-receipt evidence" in request


def test_parse_process_validator_response_json() -> None:
    parsed = _parse_process_validator_response('{"valid": false, "issues": ["unsupported"]}')
    assert parsed is not None
    assert parsed["valid"] is False


def test_validate_process_no_longer_requires_contract(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "data_agent_baseline.agents.process_validator.invoke_model_with_retries",
        lambda *args, **kwargs: AIMessage(
            content='{"valid": true, "issues": [], "required_next_actions": [], "semantic_ledger": {}}'
        ),
    )

    result = validate_process(
        model=object(),  # type: ignore[arg-type]
        question="Question?",
        answer={"columns": ["entity"], "rows": [["A"]]},
    )

    assert result["valid"] is True
    assert "validator_error" not in result


def test_validate_process_ignores_legacy_contract_field(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "data_agent_baseline.agents.process_validator.invoke_model_with_retries",
        lambda *args, **kwargs: AIMessage(
            content=json.dumps(
                {
                    "valid": True,
                    "issues": [],
                    "required_next_actions": [],
                    "semantic_ledger": {"intent_summary": "ok"},
                    "submission_contract": {"expected_columns": ["entity"]},
                }
            )
        ),
    )

    result = validate_process(
        model=object(),  # type: ignore[arg-type]
        question="Question?",
        answer={"columns": ["entity"], "rows": [["A"]]},
    )

    assert result["valid"] is True
    assert "submission_contract" not in result


def test_validate_process_keeps_technical_fail_open(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "data_agent_baseline.agents.process_validator.invoke_model_with_retries",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    result = validate_process(model=object(), question="Question?")  # type: ignore[arg-type]
    assert result["valid"] is True
    assert result["validator_error"] == "offline"


def test_supporting_source_evidence_uses_only_successful_tool_results() -> None:
    steps = [
        {
            "step_index": 3,
            "node": "tool",
            "tool_calls": [{"id": "doc", "name": "read_doc", "args": {"path": "doc/a.md"}}],
            "tool_results": [{"ok": True, "tool": "read_doc", "content": "binding fact"}],
        },
        {
            "step_index": 4,
            "node": "tool",
            "tool_calls": [{"id": "bad", "name": "read_doc", "args": {"path": "doc/b.md"}}],
            "tool_results": [{"ok": False, "tool": "read_doc", "content": "error"}],
        },
        {
            "step_index": 5,
            "node": "tool",
            "tool_calls": [{"id": "frame", "name": "read_context_image", "args": {"path": "frames/1.jpg"}}],
            "tool_results": [{"ok": True, "tool": "read_context_image", "content": "visible label"}],
        },
    ]

    evidence = _build_supporting_source_evidence(
        steps,
        {"source_tool_args": {"code": "read doc/a.md"}},
    )

    assert [item["tool"] for item in evidence["evidence_items"]] == [
        "read_doc",
        "read_context_image",
    ]
    assert evidence["schema_version"] == 2
    assert evidence["evidence_items"][0]["tool_call_id"] == "doc"
    assert evidence["evidence_items"][1]["observation"]["locator"] == {
        "frame_path": "frames/1.jpg"
    }
    assert evidence["omitted_or_unusable"] == [
        {"tool": "read_doc", "source": "doc/b.md", "reason": "not_successful"}
    ]


def test_video_timeline_and_visual_receipt_are_selected_but_summary_is_not() -> None:
    steps = [
        {
            "step_index": 1,
            "node": "tool",
            "tool_calls": [
                {"id": "timeline", "name": "read_doc", "args": {"path": "video/briefing_timeline.md"}},
            ],
            "tool_results": [{"ok": True, "tool": "read_doc", "content": "00:00 rule"}],
        },
        {
            "step_index": 2,
            "node": "tool",
            "tool_calls": [
                {"id": "summary", "name": "read_doc", "args": {"path": "video/briefing_video_summary.md"}},
            ],
            "tool_results": [{"ok": True, "tool": "read_doc", "content": "narrative context"}],
        },
        {
            "step_index": 3,
            "node": "tool",
            "tool_calls": [
                {"id": "frame", "name": "read_context_image", "args": {"path": "video/frame_001.jpg"}},
            ],
            "tool_results": [{"ok": True, "tool": "read_context_image", "content": {"status": "image attached"}}],
        },
        {
            "step_index": 4,
            "node": "tool",
            "tool_calls": [
                {"id": "receipt", "name": "record_visual_evidence", "args": {"path": "video/frame_001.jpg", "observations": "threshold: 100"}},
            ],
            "tool_results": [{"ok": True, "tool": "record_visual_evidence", "content": {"path": "video/frame_001.jpg", "observations": "threshold: 100"}}],
        },
    ]

    evidence = _build_supporting_source_evidence(steps, {"source_tool_args": {"sql": "SELECT 1"}})

    assert [item["tool"] for item in evidence["evidence_items"]] == [
        "read_doc",
        "read_doc",
        "read_context_image",
        "record_visual_evidence",
    ]
    # timeline is first, summary is second
    assert evidence["evidence_items"][0]["source"]["path"] == "video/briefing_timeline.md"
    assert evidence["evidence_items"][1]["source"]["path"] == "video/briefing_video_summary.md"
    assert evidence["evidence_items"][1]["capabilities"] == ["video_narrative_context"]
    assert evidence["evidence_items"][3]["capabilities"] == ["visual_fact_receipt"]
    assert evidence["omitted_or_unusable"] == []


def test_recent_trace_links_success_evidence_without_repeating_result_content() -> None:
    steps = [
        {
            "step_index": 8,
            "node": "model",
            "ok": True,
            "assistant_message": "unneeded model text",
            "model_response": {
                "finish_reason": "tool_calls",
                "tool_call_names": ["read_doc", "read_context_image", "execute_probe_query"],
                "content_preview": "called tools",
                "reasoning_content": "REASONING_CONTENT_MUST_NOT_LEAK",
            },
        },
        {
            "step_index": 9,
            "node": "tool",
            "ok": False,
            "tool_calls": [
                {"id": "doc", "name": "read_doc", "args": {"path": "doc/a.md"}},
                {
                    "id": "frame",
                    "name": "read_context_image",
                    "args": {"path": "frames/1.jpg"},
                },
                {
                    "id": "bad_sql",
                    "name": "execute_probe_query",
                    "args": {"queries": ["SELECT broken FROM table"]},
                },
            ],
            "tool_results": [
                {"ok": True, "tool": "read_doc", "content": "SUCCESS_DOCUMENT_FACT"},
                {"ok": True, "tool": "read_context_image", "content": "SUCCESS_VISUAL_FACT"},
                {"ok": False, "tool": "execute_probe_query", "content": {"error": "BROKEN_QUERY"}},
            ],
        },
    ]
    evidence = _build_supporting_source_evidence(
        steps,
        {"source_tool_args": {"code": "read doc/a.md"}},
    )
    request = _build_process_validation_request(
        question="Question?",
        supporting_source_evidence=evidence,
        recent_steps=steps,
    )

    recent_json = request.split("## Recent Trace Steps\n```json\n", 1)[1].split("\n```", 1)[0]
    recent_steps = json.loads(recent_json)
    tool_step = recent_steps[1]

    assert request.count("SUCCESS_DOCUMENT_FACT") == 2  # once in evidence, once in trace content
    assert request.count("SUCCESS_VISUAL_FACT") == 2
    assert "REASONING_CONTENT_MUST_NOT_LEAK" not in request
    assert tool_step["tool_calls"][0]["args"] == {"path": "doc/a.md"}
    assert tool_step["tool_results"] == [
        {
            "tool": "read_doc",
            "ok": True,
            "supporting_evidence_id": "step_9:call_doc",
            "content": "SUCCESS_DOCUMENT_FACT",
        },
        {
            "tool": "read_context_image",
            "ok": True,
            "supporting_evidence_id": "step_9:call_frame",
            "content": "SUCCESS_VISUAL_FACT",
        },
        {"tool": "execute_probe_query", "ok": False, "error": "BROKEN_QUERY"},
    ]
    assert evidence["omitted_or_unusable"] == [
        {
            "tool": "execute_probe_query",
            "source": "execute_probe_query",
            "reason": "not_successful",
        }
    ]
