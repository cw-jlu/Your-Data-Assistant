from __future__ import annotations

from data_agent_baseline.agents.answer_validator import _build_validation_request
from data_agent_baseline.agents.langgraph_runtime import _build_answer_validator_context


def test_answer_validator_context_omits_row_samples() -> None:
    rows = [[None], ["2024-01-01"], ["2024-01-02"], ["2024-01-03"]] + [
        [f"2024-02-{index:02d}"] for index in range(1, 15)
    ]
    _, overview, bounded = _build_answer_validator_context(
        {"columns": ["value"], "rows": rows},
        max_str_tokens=100,
        max_list_items=50,
        distinct_examples_per_column=5,
    )

    assert overview["row_count"] == 18
    assert overview["row_length_counts"] == {"1": 18}
    assert "row_index" not in str(overview)
    assert "head_rows" not in str(overview)
    assert bounded is True


def test_answer_validator_context_counts_duplicate_complete_rows() -> None:
    _, overview, _ = _build_answer_validator_context(
        {"columns": ["entity", "value"], "rows": [["A", 1], ["A", 1], ["A", 2]]},
        max_str_tokens=100,
        max_list_items=50,
    )

    assert overview["duplicate_row_count"] == 1


def test_validation_request_includes_video_ranking_evidence_only_when_supplied() -> None:
    video_evidence = {
        "schema_version": 2,
        "evidence_items": [
            {
                "tool": "read_doc",
                "capabilities": ["document_text_fact"],
                "source": {"path": "video/briefing_timeline.md"},
            },
            {
                "tool": "read_context_image",
                "capabilities": ["visual_fact"],
                "source": {"path": "video/stable_006.jpg"},
                "observation": {"locator": {"frame_path": "video/stable_006.jpg"}},
            },
            {
                "tool": "record_visual_evidence",
                "capabilities": ["visual_fact_receipt"],
                "source": {"path": "video/stable_006.jpg"},
                "observation": {"evidence_excerpt": "Report scope: Top 3"},
            },
        ],
        "omitted_or_unusable": [],
    }
    request = _build_validation_request(
        question="List entities.",
        answer={"columns": ["entity"], "rows": [["A"]]},
        answer_structure_overview={"columns": ["entity"], "row_count": 1},
        supporting_source_evidence=video_evidence,
    )

    assert "Supporting Source Evidence" in request
    assert "Report scope: Top 3" in request
    assert "Programmatic Submission Risk Report" not in request
    assert "Submitted Answer Structure Overview" in request


def test_validation_request_omits_video_evidence_when_none_is_supplied() -> None:
    request = _build_validation_request(
        question="List entities.",
        answer={"columns": ["entity"], "rows": [["A"]]},
    )

    assert "Supporting Source Evidence" not in request


def test_validation_request_includes_knowledge_docs_for_identifier_rules() -> None:
    request = _build_validation_request(
        question="List records.",
        answer={"columns": ["value"], "rows": [[1]]},
        knowledge_docs=[
            {
                "path": "knowledge.md",
                "kind": "document",
                "content": "table: sample\nprimary_key: RecordNo\nfield: value",
            }
        ],
    )

    assert "Knowledge Documents (authoritative field definitions)" in request
    assert "primary_key: RecordNo" in request
    assert "never treat the missing identifier as permission to deduplicate" in request


def test_validation_request_includes_submission_field_context() -> None:
    request = _build_validation_request(
        question="List records.",
        answer={"columns": ["shareholder_name"], "rows": [["A"]]},
        submission_field_context={
            "status": "complete",
            "field_universe": [
                {
                    "table": "lc_sharefp",
                    "alias": "lc_sharefp",
                    "fields": [{"name": "FPSHName", "qualified_name": "lc_sharefp.FPSHName"}],
                }
            ],
            "output_lineage": [
                {
                    "output_column": "shareholder_name",
                    "sources": [
                        {
                            "table": "lc_sharefp",
                            "alias": "lc_sharefp",
                            "field": "FPSHName",
                            "qualified_name": "lc_sharefp.FPSHName",
                        }
                    ],
                }
            ],
        },
    )

    assert "Submission Field Context" in request
    assert "field_universe" in request
    assert "shareholder_name" in request
    assert "lc_sharefp.FPSHName" in request


def test_validation_request_explains_context_truncation_is_not_answer_truncation() -> None:
    request = _build_validation_request(
        question="List records.",
        answer={"columns": ["value"], "rows": [[1]]},
        answer_truncated=True,
        answer_row_count=1,
        preview_row_limit=50,
    )

    assert "runtime retains and scores the complete submitted answer separately" in request
    assert "They are not evidence that the submitted answer omitted rows" in request
    assert "Never reject an answer solely because of these context-bound signals" in request
    assert "Tool evidence and the answer structure overview must not be used" in request
