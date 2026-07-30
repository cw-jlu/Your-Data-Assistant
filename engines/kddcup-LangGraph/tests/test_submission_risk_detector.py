from __future__ import annotations

from data_agent_baseline.agents.submission_risk_detector import detect_submission_risks


def _kinds(report: dict[str, object]) -> set[str]:
    return {
        str(detection["kind"])
        for detection in report["detected"]
        if isinstance(detection, dict)
    }


def test_detects_sql_source_risks_with_sqlglot_ast() -> None:
    report = detect_submission_risks(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {
                "queries": [
                    "SELECT DISTINCT value FROM t WHERE value IS NOT NULL LIMIT 10",
                    "SELECT group_id, COUNT(DISTINCT value) FROM t GROUP BY group_id",
                    "SELECT value FROM t WHERE TRIM(value) != '' FETCH FIRST 5 ROWS ONLY",
                ]
            },
        }
    )

    assert {"null_filter", "row_limit", "deduplication", "row_collapse"} <= _kinds(report)
    assert report["has_high_confidence_risks"] is True
    assert report["parse_errors"] == []


def test_sql_detector_ignores_comments_and_strings_for_regex_fallback() -> None:
    report = detect_submission_risks(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {
                "queries": [
                    "SELECT '-- WHERE x IS NOT NULL LIMIT 5' AS note /* SELECT DISTINCT x */"
                ]
            },
        }
    )

    assert _kinds(report) == set()


def test_sql_detector_falls_back_when_parse_fails() -> None:
    report = detect_submission_risks(
        {
            "source_tool": "execute_probe_query",
            "source_tool_args": {"queries": ["SELECT TOP 5 value FROM t"]},
        }
    )

    assert "row_limit" in _kinds(report)
    assert report["parse_errors"]


def test_detects_python_ast_risks_and_embedded_sql() -> None:
    report = detect_submission_risks(
        {
            "source_tool": "execute_python",
            "source_tool_args": {
                "code": "\n".join(
                    [
                        "rows = query(\"SELECT DISTINCT value FROM t WHERE value IS NOT NULL\")",
                        "df = df.dropna().drop_duplicates().groupby('value').size()",
                        "limited = df.iloc[:10]",
                        "values = set(limited['value'].unique())",
                    ]
                )
            },
        }
    )

    assert {"null_filter", "row_limit", "deduplication", "row_collapse"} <= _kinds(report)
    assert report["has_high_confidence_risks"] is True
    assert report["parse_errors"] == []


def test_python_detector_falls_back_when_parse_fails() -> None:
    report = detect_submission_risks(
        {
            "source_tool": "execute_python",
            "source_tool_args": {"code": "df = df.dropna(\nlimited = df.head(5)"},
        }
    )

    assert {"null_filter", "row_limit"} <= _kinds(report)
    assert report["parse_errors"]
