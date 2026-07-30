from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import duckdb
import pytest
from langchain_core.messages import AIMessage

from data_agent_baseline.agents.langgraph_runtime import LangGraphAgent, LangGraphAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import (
    DataInspectorConfig,
    DataInspectorSampleBudget,
    DataInspectorSemanticViewConfig,
    load_app_config,
)
from data_agent_baseline.inspectors.data_understanding_agent import DataUnderstandingAgent

from data_agent_baseline.inspectors.semantic_catalog import (
    build_lightweight_catalog,
    build_semantic_catalog,
    iter_logical_tables,
)
from data_agent_baseline.inspectors.semantic_views import build_derived_views
from data_agent_baseline.tools.duckdb_schema import create_duckdb_views, validate_derived_views
from data_agent_baseline.run.runner import _write_task_outputs
from data_agent_baseline.tools.registry import create_default_tool_registry


def _create_task(
    tmp_path: Path, question: str = "What's the finish time for the ranked second driver?"
) -> PublicTask:
    task_dir = tmp_path / "task_demo"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "results.csv").write_text(
        "raceId,driverId,rank,position,time\n1,10,2,1,1:30.000\n",
        encoding="utf-8",
    )
    (context_dir / "bad.json").write_text("{not json", encoding="utf-8")
    (context_dir / "data.json").write_text(
        json.dumps([{"raceId": 1, "name": "Chinese Grand Prix", "year": 2008}]),
        encoding="utf-8",
    )
    (context_dir / "knowledge.md").write_text(
        "# Notes\nrank differs from position\n", encoding="utf-8"
    )
    db_path = context_dir / "sample.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE races (raceId INTEGER, name TEXT)")
        conn.execute("INSERT INTO races VALUES (1, 'Chinese Grand Prix')")
    return PublicTask(
        record=TaskRecord(task_id="task_demo", difficulty="easy", question=question),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _create_cost_event_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_25_like"
    context_dir = task_dir / "context"
    (context_dir / "json").mkdir(parents=True)
    (context_dir / "csv").mkdir()
    (context_dir / "json" / "event.json").write_text(
        json.dumps(
            {
                "table": "event",
                "records": [
                    {"event_id": "event_a", "event_name": "November Speaker"},
                    {"event_id": "event_b", "event_name": "October Speaker"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (context_dir / "json" / "expense.json").write_text(
        json.dumps(
            {
                "table": "expense",
                "records": [
                    {"expense_id": "exp_a", "cost": 6.0, "link_to_budget": "budget_a"},
                    {"expense_id": "exp_b", "cost": 6.0, "link_to_budget": "budget_b"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (context_dir / "csv" / "budget.csv").write_text(
        "budget_id,amount,spent,link_to_event\nbudget_a,10,6,event_a\nbudget_b,10,6,event_b\n",
        encoding="utf-8",
    )
    (context_dir / "knowledge.md").write_text(
        "# Financials\n"
        "- amount: budgeted amount, not actual cost.\n"
        "- spent: total expenditure associated with a budget.\n"
        "- cost: individual expense amount.\n",
        encoding="utf-8",
    )
    return PublicTask(
        record=TaskRecord(
            task_id="task_25_like", difficulty="easy", question="Which event has the lowest cost?"
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


class ScriptedToolCallingModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.invocations = []

    def bind_tools(self, tools, tool_choice="auto", parallel_tool_calls=False):  # noqa: ANN001
        del tools, tool_choice, parallel_tool_calls
        return self

    def invoke(self, messages):  # noqa: ANN001
        self.invocations.append(messages)
        if not self._responses:
            raise RuntimeError("No scripted responses remaining.")
        return self._responses.pop(0)


def test_semantic_catalog_handles_supported_assets_and_bad_json(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    (task.context_dir / "broken.db").write_text("not a sqlite database", encoding="utf-8")

    catalog = build_semantic_catalog(
        task,
        budget=DataInspectorSampleBudget(max_doc_tokens=40),
    )

    asset_paths = {asset["asset_path"] for asset in catalog["assets"]}
    assert {
        "results.csv",
        "data.json",
        "knowledge.md",
        "sample.db",
        "bad.json",
        "broken.db",
    } <= asset_paths
    assert any(
        schema["kind"] == "csv" and schema["asset_path"] == "results.csv"
        for schema in catalog["schemas"]
    )
    assert any(
        schema["kind"] == "sqlite" and schema["asset_path"] == "sample.db"
        for schema in catalog["schemas"]
    )
    sqlite_schema = next(
        schema for schema in catalog["schemas"] if schema["asset_path"] == "sample.db"
    )
    races_table = next(table for table in sqlite_schema["tables"] if table["name"] == "races")
    race_id_field = next(field for field in races_table["fields"] if field["name"] == "raceId")
    name_field = next(field for field in races_table["fields"] if field["name"] == "name")
    assert race_id_field["distinct_values"] == [{"value": "1", "count": 1}]
    assert name_field["distinct_values"] == [{"value": "Chinese Grand Prix", "count": 1}]
    assert any(item["asset_path"] == "bad.json" for item in catalog["semantic_uncertainties"])
    assert any(item["asset_path"] == "broken.db" for item in catalog["semantic_uncertainties"])
    assert catalog["relationships"] == []
    recommended_tools = {
        tool_name for asset in catalog["assets"] for tool_name in asset.get("recommended_tools", [])
    }
    assert {"read_csv", "read_json", "inspect_all_schema", "inspect_sqlite_schema"}.isdisjoint(
        recommended_tools
    )


def test_langgraph_agent_data_inspector_failure_does_not_block_answer(
    tmp_path: Path, monkeypatch
) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )

    def fail_explore(self, *, context_dir, task_id=""):
        raise RuntimeError("synthetic inspector failure")

    monkeypatch.setattr(
        "data_agent_baseline.inspectors.data_understanding_agent.DataUnderstandingAgent.explore_data_globally",
        fail_explore,
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert len(result.steps) > 0
    first_step = result.steps[0].to_dict()
    assert first_step["node"] == "global_data_exploration"
    assert first_step["ok"] is False


def test_langgraph_agent_records_global_exploration_failure_and_continues(
    tmp_path: Path, monkeypatch
) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    def fail_explore(self, *, context_dir, task_id=""):
        raise RuntimeError("synthetic profiling failure")

    monkeypatch.setattr(
        "data_agent_baseline.inspectors.data_understanding_agent.DataUnderstandingAgent.explore_data_globally",
        fail_explore,
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    first_step = result.steps[0].to_dict()
    assert first_step["node"] == "global_data_exploration"
    assert first_step["ok"] is False
    assert "synthetic profiling failure" in first_step["tool_results"][0]["error"]


def test_langgraph_agent_receive_problem_injects_catalog(tmp_path: Path) -> None:
    task = _create_cost_event_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["event_name"], "rows": [["November Speaker"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(),
        ),
    )

    result = agent.run(task)

    assert result.succeeded is True
    first_request = model.invocations[-1]
    injected_messages = [getattr(message, "content", "") for message in first_request]
    injected_text = "\n".join(str(content) for content in injected_messages)
    assert "lightweight catalog" in injected_text.lower()


def test_runner_writes_inspector_artifacts(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    run_result = {
        "task_id": "task_demo",
        "answer": None,
        "steps": [],
        "failure_reason": "none",
        "succeeded": False,
        "inspector": {
            "semantic_catalog": {"assets": []},
            "semantic_index": {"mode": "keyword"},
            "data_understanding_handoff": {"brief_markdown": "ok"},
            "handoff_status": "complete",
            "validation_errors": [],
            "inspector_steps": [{"phase": "deterministic_context"}],
            "global_data_profile": "# Test Profile\n\nSome profile content.",
        },
    }

    _write_task_outputs("task_demo", run_output_dir, run_result)

    assert (run_output_dir / "task_demo" / "global_data_profile.json").exists()
    assert (run_output_dir / "task_demo" / "semantic_catalog.json").exists()
    assert (run_output_dir / "task_demo" / "semantic_index.json").exists()
    assert (run_output_dir / "task_demo" / "data_understanding_handoff.json").exists()


def test_runner_writes_global_profile_from_top_level_result(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    run_result = {
        "task_id": "task_demo",
        "answer": None,
        "steps": [],
        "failure_reason": "stopped during grounding",
        "succeeded": False,
        "inspector": None,
        "global_data_profile": "## Global Data Profile\n\nRecovered from stage 1.",
    }

    _write_task_outputs("task_demo", run_output_dir, run_result)

    profile_path = run_output_dir / "task_demo" / "global_data_profile.json"
    assert (
        profile_path.read_text(encoding="utf-8")
        == "## Global Data Profile\n\nRecovered from stage 1."
    )


def test_runner_preserves_global_profile_from_partial_trace(tmp_path: Path) -> None:
    run_output_dir = tmp_path / "run"
    task_dir = run_output_dir / "task_demo"
    task_dir.mkdir(parents=True)
    (task_dir / "trace.json").write_text(
        json.dumps(
            {
                "task_id": "task_demo",
                "answer": None,
                "steps": [{"node": "global_data_exploration"}],
                "failure_reason": None,
                "succeeded": False,
                "inspector": None,
                "partial": True,
                "global_data_profile": "## Global Data Profile\n\nPreserved from live trace.",
            }
        ),
        encoding="utf-8",
    )
    run_result = {
        "task_id": "task_demo",
        "answer": None,
        "steps": [],
        "failure_reason": "Task failed before final state.",
        "succeeded": False,
        "inspector": None,
    }

    _write_task_outputs("task_demo", run_output_dir, run_result)

    profile_path = task_dir / "global_data_profile.json"
    trace_payload = json.loads((task_dir / "trace.json").read_text(encoding="utf-8"))
    assert (
        profile_path.read_text(encoding="utf-8")
        == "## Global Data Profile\n\nPreserved from live trace."
    )
    assert (
        trace_payload["global_data_profile"]
        == "## Global Data Profile\n\nPreserved from live trace."
    )
    assert trace_payload["finalized_from_partial_trace"] is True


def test_explore_data_globally_returns_lightweight_catalog(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    (doc_dir / "mf_investadvisoroutline.md").write_text("# Advisor Outline\n", encoding="utf-8")
    profile, _catalog = DataUnderstandingAgent(
        config=DataInspectorConfig(sample_budget=DataInspectorSampleBudget(max_doc_tokens=10)),
    ).explore_data_globally(context_dir=task.context_dir, task_id=task.task_id)

    payload = json.loads(profile)
    # Lightweight catalog has task_id, query_surfaces, documents/media, knowledge_documents
    assert "task_id" in payload
    assert "query_surfaces" in payload
    assert "structured_tables" not in payload
    assert "semantic_views" not in payload
    assert "documents" in payload
    assert "media" in payload
    assert "knowledge_documents" in payload
    assert "relationships" not in payload
    assert "instructions" not in payload
    assert "phase" not in payload
    doc_entry = next(
        doc for doc in payload["documents"] if doc["path"] == "doc/mf_investadvisoroutline.md"
    )
    assert doc_entry["kind"] == "document"
    assert doc_entry["stem"] == "mf_investadvisoroutline"
    assert doc_entry["recommended_tools"] == ["search_doc", "read_doc"]
    assert not any(
        surface["table"] == "mf_investadvisoroutline" for surface in payload["query_surfaces"]
    )
    # Check query surfaces stay compact and do not include full field profiles.
    for surface in payload["query_surfaces"]:
        assert "kind" in surface
        assert "key_columns" in surface
        assert "columns" not in surface
        assert "distinct_values" not in surface
        assert "cardinality" not in surface


def test_semantic_views_can_be_disabled_by_config(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_semantic_views_off"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "sales.csv").write_text(
        "CompanyCode,Amount\n1,10\n1,5\n2,20\n3,30\n",
        encoding="utf-8",
    )
    db_path = context_dir / "sample.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE lc_exgindustry (CompanyCode INTEGER, SecondIndustryName TEXT)"
        )
        conn.execute("INSERT INTO lc_exgindustry VALUES (1, 'A'), (2, 'B'), (3, 'C')")
    task = PublicTask(
        record=TaskRecord(task_id="task_semantic_views_off", difficulty="easy", question="sales by industry"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(
        task,
        budget=DataInspectorSampleBudget(),
        semantic_view_config=DataInspectorSemanticViewConfig(enabled=False),
    )
    lightweight = build_lightweight_catalog(catalog)

    assert catalog["derived_views"] == []
    assert all(surface["kind"] == "original_table" for surface in lightweight["query_surfaces"])
    assert {surface["table"] for surface in lightweight["query_surfaces"]} == {
        "sales",
        "lc_exgindustry",
    }


def test_semantic_views_are_exposed_in_lightweight_catalog(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_semantic_views"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "sales.csv").write_text(
        "CompanyCode,Amount\n1,10\n1,5\n2,20\n3,30\n",
        encoding="utf-8",
    )
    db_path = context_dir / "sample.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE lc_exgindustry (CompanyCode INTEGER, SecondIndustryName TEXT)"
        )
        conn.execute("INSERT INTO lc_exgindustry VALUES (1, 'A'), (2, 'B'), (3, 'C')")
    task = PublicTask(
        record=TaskRecord(task_id="task_semantic_views", difficulty="easy", question="sales by industry"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(
        task,
        budget=DataInspectorSampleBudget(),
        semantic_view_config=DataInspectorSemanticViewConfig(),
    )
    lightweight = build_lightweight_catalog(catalog)

    view = next(view for view in catalog["derived_views"] if view["name"] == "v_sales_enriched")
    assert view["kind"] == "derived_view"
    assert view["base_table"] == "sales"
    assert view["grain"] == "same_as_base_table"
    assert view["is_original_table"] is False
    assert all(column["source_table"] and column["source_field"] for column in view["columns"])
    assert any(column["name"] == "SecondIndustryName" for column in view["columns"])
    sales_surfaces = [
        item for item in lightweight["query_surfaces"] if item["base_table"] == "sales"
    ]
    assert len(sales_surfaces) == 1
    assert sales_surfaces[0]["table"] == "v_sales_enriched"
    assert sales_surfaces[0]["kind"] == "derived_view"
    assert sales_surfaces[0]["is_original_table"] is False
    assert sales_surfaces[0]["join_status"] == "enriched"
    assert any(
        dimension["table"] == "lc_exgindustry"
        for dimension in sales_surfaces[0]["attached_dimensions"]
    )
    assert "sales" not in {item["table"] for item in lightweight["query_surfaces"]}
    assert any(
        item["table"] == "lc_exgindustry" and item["kind"] == "original_table"
        for item in lightweight["query_surfaces"]
    )


def test_semantic_view_json_records_join_fields_use_logical_column_names(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_json_semantic_view"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "posts.json").write_text(
        json.dumps(
            {
                "records": [
                    {"Id": 10, "OwnerUserId": 1},
                    {"Id": 11, "OwnerUserId": 2},
                    {"Id": 12, "OwnerUserId": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    db_path = context_dir / "users.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE users (Id INTEGER PRIMARY KEY, DisplayName TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        conn.execute("INSERT INTO users VALUES (2, 'Bob')")
    task = PublicTask(
        record=TaskRecord(
            task_id="task_json_semantic_view",
            difficulty="easy",
            question="post owners",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())

    view = next(view for view in catalog["derived_views"] if view["name"] == "v_posts_enriched")
    assert view["joins"][0]["source_fields"] == ["OwnerUserId"]
    assert view["joins"][0]["target_fields"] == ["Id"]
    assert any(column["name"] == "DisplayName" for column in view["columns"])

    conn = duckdb.connect(":memory:")
    try:
        create_duckdb_views(
            conn,
            context_dir,
            catalog,
            logical_tables=iter_logical_tables(catalog),
            sql="SELECT DisplayName FROM v_posts_enriched",
        )
        rows = conn.execute(
            "SELECT Id, DisplayName FROM v_posts_enriched ORDER BY Id"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [(10, "Alice"), (11, "Bob"), (12, "Alice")]


def test_validate_derived_views_drops_unrenderable_view(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "sales.csv").write_text("CompanyCode,Amount\n1,10\n", encoding="utf-8")
    catalog = {
        "schemas": [{"asset_path": "sales.csv", "kind": "csv"}],
        "derived_views": [
            {
                "name": "v_sales_enriched",
                "base_table": "sales",
                "columns": [
                    {
                        "name": "Missing",
                        "type": "unknown",
                        "source_table": "sales",
                        "source_field": "Missing",
                    }
                ],
                "joins": [],
            }
        ],
    }
    logical_tables = [
        {
            "table": "sales",
            "source_asset_path": "sales.csv",
            "source_kind": "csv",
            "columns": [{"name": "CompanyCode"}, {"name": "Amount"}],
        }
    ]

    valid_views, warnings = validate_derived_views(
        context_dir,
        catalog,
        logical_tables=logical_tables,
    )

    assert valid_views == []
    assert any("derived_view_validation_failed:v_sales_enriched" in warning for warning in warnings)


def test_validate_derived_views_drops_fanout_view(tmp_path: Path) -> None:
    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "sales.csv").write_text(
        "CompanyCode,Amount\n1,10\n2,20\n",
        encoding="utf-8",
    )
    (context_dir / "industry.csv").write_text(
        "CompanyCode,Industry\n1,A\n1,B\n2,C\n",
        encoding="utf-8",
    )
    catalog = {
        "schemas": [
            {"asset_path": "sales.csv", "kind": "csv"},
            {"asset_path": "industry.csv", "kind": "csv"},
        ],
        "derived_views": [
            {
                "name": "v_sales_enriched",
                "base_table": "sales",
                "columns": [
                    {
                        "name": "CompanyCode",
                        "type": "unknown",
                        "source_table": "sales",
                        "source_field": "CompanyCode",
                    },
                    {
                        "name": "Industry",
                        "type": "unknown",
                        "source_table": "industry",
                        "source_field": "Industry",
                    },
                ],
                "joins": [
                    {
                        "dimension_table": "industry",
                        "source_fields": ["CompanyCode"],
                        "target_fields": ["CompanyCode"],
                    }
                ],
            }
        ],
    }
    logical_tables = [
        {
            "table": "sales",
            "source_asset_path": "sales.csv",
            "source_kind": "csv",
            "columns": [{"name": "CompanyCode"}, {"name": "Amount"}],
        },
        {
            "table": "industry",
            "source_asset_path": "industry.csv",
            "source_kind": "csv",
            "columns": [{"name": "CompanyCode"}, {"name": "Industry"}],
        },
    ]

    valid_views, warnings = validate_derived_views(
        context_dir,
        catalog,
        logical_tables=logical_tables,
    )

    assert valid_views == []
    assert warnings == ["derived_view_fanout:v_sales_enriched:base_rows=2:view_rows=3"]


def _semantic_view_logical_table(
    table: str,
    *,
    row_count: int,
    columns: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "table": table,
        "source_asset_path": f"{table}.csv",
        "source_kind": "csv",
        "row_count": row_count,
        "columns": columns,
    }


def _semantic_view_relationship(
    *,
    source_table: str,
    source_field: str,
    target_table: str,
    target_field: str,
    relationship_type: str = "foreign_key",
    cardinality: str = "many_to_one",
    confidence: float = 0.99,
    matched_source_distinct_ratio: float = 1.0,
    matched_source_row_ratio: float = 1.0,
    target_uniqueness_ratio: float = 1.0,
) -> dict[str, object]:
    return {
        "source": {
            "asset_path": f"{source_table}.csv",
            "table": None,
            "fields": [source_field],
        },
        "target": {
            "asset_path": f"{target_table}.csv",
            "table": None,
            "fields": [target_field],
        },
        "relationship_type": relationship_type,
        "cardinality": cardinality,
        "confidence": confidence,
        "evidence": {
            "matched_source_distinct_ratio": matched_source_distinct_ratio,
            "matched_source_row_ratio": matched_source_row_ratio,
            "target_uniqueness_ratio": target_uniqueness_ratio,
        },
    }


def test_semantic_view_planner_uses_relationship_evidence_not_table_names() -> None:
    source = _semantic_view_logical_table(
        "a",
        row_count=5,
        columns=[
            {"name": "x", "type": "BIGINT", "missing_count": 0, "cardinality": 3},
            {"name": "m", "type": "DOUBLE", "missing_count": 0, "cardinality": 5},
        ],
    )
    target = _semantic_view_logical_table(
        "b",
        row_count=3,
        columns=[
            {"name": "y", "type": "BIGINT", "missing_count": 0, "cardinality": 3},
            {"name": "z", "type": "VARCHAR", "missing_count": 0, "cardinality": 3},
        ],
    )
    views = build_derived_views(
        {"relationships": [_semantic_view_relationship(
            source_table="a",
            source_field="x",
            target_table="b",
            target_field="y",
            relationship_type="same_key",
        )]},
        logical_tables=[source, target],
        config=DataInspectorSemanticViewConfig(),
    )

    assert [view["name"] for view in views] == ["v_a_enriched"]
    assert views[0]["joins"][0]["dimension_table"] == "b"
    assert any(column["name"] == "z" and column["role"] == "dimension" for column in views[0]["columns"])


def test_semantic_view_planner_rejects_non_unique_target_even_with_dimension_like_name() -> None:
    source = _semantic_view_logical_table(
        "sales",
        row_count=5,
        columns=[{"name": "x", "type": "BIGINT", "missing_count": 0, "cardinality": 3}],
    )
    target = _semantic_view_logical_table(
        "exgindustry",
        row_count=6,
        columns=[
            {"name": "y", "type": "BIGINT", "missing_count": 0, "cardinality": 3},
            {"name": "z", "type": "VARCHAR", "missing_count": 0, "cardinality": 3},
        ],
    )

    views = build_derived_views(
        {"relationships": [_semantic_view_relationship(
            source_table="sales",
            source_field="x",
            target_table="exgindustry",
            target_field="y",
            target_uniqueness_ratio=0.5,
        )]},
        logical_tables=[source, target],
        config=DataInspectorSemanticViewConfig(),
    )

    assert views == []


def test_semantic_view_planner_allows_safe_lookup_even_with_fact_like_name() -> None:
    source = _semantic_view_logical_table(
        "events",
        row_count=5,
        columns=[{"name": "x", "type": "BIGINT", "missing_count": 0, "cardinality": 3}],
    )
    target = _semantic_view_logical_table(
        "dailyquote",
        row_count=3,
        columns=[
            {"name": "y", "type": "BIGINT", "missing_count": 0, "cardinality": 3},
            {"name": "z", "type": "VARCHAR", "missing_count": 0, "cardinality": 2},
        ],
    )

    views = build_derived_views(
        {"relationships": [_semantic_view_relationship(
            source_table="events",
            source_field="x",
            target_table="dailyquote",
            target_field="y",
        )]},
        logical_tables=[source, target],
        config=DataInspectorSemanticViewConfig(),
    )

    assert [view["name"] for view in views] == ["v_events_enriched"]
    assert views[0]["joins"][0]["dimension_table"] == "dailyquote"


def test_semantic_view_planner_skips_high_cardinality_numeric_payload() -> None:
    source = _semantic_view_logical_table(
        "a",
        row_count=100,
        columns=[{"name": "x", "type": "BIGINT", "missing_count": 0, "cardinality": 10}],
    )
    target = _semantic_view_logical_table(
        "b",
        row_count=100,
        columns=[
            {"name": "y", "type": "BIGINT", "missing_count": 0, "cardinality": 10},
            {"name": "raw_metric", "type": "DOUBLE", "missing_count": 0, "cardinality": 100},
            {"name": "z", "type": "VARCHAR", "missing_count": 0, "cardinality": 3},
        ],
    )

    views = build_derived_views(
        {"relationships": [_semantic_view_relationship(
            source_table="a",
            source_field="x",
            target_table="b",
            target_field="y",
        )]},
        logical_tables=[source, target],
        config=DataInspectorSemanticViewConfig(),
    )

    dimension_columns = [column["name"] for column in views[0]["columns"] if column["role"] == "dimension"]
    assert "z" in dimension_columns
    assert "raw_metric" not in dimension_columns


def test_semantic_view_planner_skips_one_to_one_by_default() -> None:
    source = _semantic_view_logical_table(
        "a",
        row_count=3,
        columns=[{"name": "x", "type": "BIGINT", "missing_count": 0, "cardinality": 3}],
    )
    target = _semantic_view_logical_table(
        "b",
        row_count=3,
        columns=[
            {"name": "y", "type": "BIGINT", "missing_count": 0, "cardinality": 3},
            {"name": "z", "type": "VARCHAR", "missing_count": 0, "cardinality": 3},
        ],
    )
    relationship = _semantic_view_relationship(
        source_table="a",
        source_field="x",
        target_table="b",
        target_field="y",
        cardinality="one_to_one",
    )

    assert build_derived_views(
        {"relationships": [relationship]},
        logical_tables=[source, target],
        config=DataInspectorSemanticViewConfig(),
    ) == []
    assert build_derived_views(
        {"relationships": [relationship]},
        logical_tables=[source, target],
        config=DataInspectorSemanticViewConfig(allow_one_to_one_enrichment=True),
    )


def test_csv_schema_includes_cardinality_and_distinct_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "id,operation,value\n"
        "1,VYBER,100\n"
        "2,VKLAD,200\n"
        "3,VYBER,300\n"
        "4,PREVOD,400\n"
        "5,VYBER,500\n"
        "6,VKLAD,600\n",
        encoding="utf-8",
    )
    from data_agent_baseline.inspectors.semantic_catalog import _read_csv_schema

    budget = DataInspectorSampleBudget()
    schema = _read_csv_schema(csv_path, "test.csv", budget)

    assert schema["row_count"] == 6
    id_field = next(f for f in schema["fields"] if f["name"] == "id")
    op_field = next(f for f in schema["fields"] if f["name"] == "operation")
    val_field = next(f for f in schema["fields"] if f["name"] == "value")

    assert id_field["cardinality"] == 6
    assert id_field["distinct_values"] == [{"value": str(i), "count": 1} for i in range(1, 7)]

    assert op_field["cardinality"] == 3
    assert sorted(item["value"] for item in op_field["distinct_values"]) == [
        "PREVOD",
        "VKLAD",
        "VYBER",
    ]
    assert val_field["cardinality"] == 6


def test_csv_schema_reports_cardinality_and_top_distinct_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "high_card.csv"
    rows = ["id,name"]
    for i in range(250):
        rows.append(f"{i},name_{i}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    from data_agent_baseline.inspectors.semantic_catalog import _read_csv_schema

    budget = DataInspectorSampleBudget()
    schema = _read_csv_schema(csv_path, "high_card.csv", budget)

    id_field = next(f for f in schema["fields"] if f["name"] == "id")
    name_field = next(f for f in schema["fields"] if f["name"] == "name")

    assert id_field["cardinality"] == 250
    assert len(id_field["distinct_values"]) == 50
    assert name_field["cardinality"] == 250
    assert len(name_field["distinct_values"]) == 50


def test_sqlite_schema_includes_cardinality_and_distinct_values(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE ops (id INTEGER, operation TEXT, value REAL)")
        conn.execute("INSERT INTO ops VALUES (1, 'VYBER', 100)")
        conn.execute("INSERT INTO ops VALUES (2, 'VKLAD', 200)")
        conn.execute("INSERT INTO ops VALUES (3, 'VYBER', 300)")
        conn.execute("INSERT INTO ops VALUES (4, 'PREVOD', 400)")
        conn.execute("INSERT INTO ops VALUES (5, 'VYBER', 500)")

    from data_agent_baseline.inspectors.semantic_catalog import _read_sqlite_schema

    budget = DataInspectorSampleBudget()
    schema = _read_sqlite_schema(db_path, "test.db", budget)

    table = schema["tables"][0]
    assert table["name"] == "ops"

    op_field = next(f for f in table["fields"] if f["name"] == "operation")
    assert op_field["cardinality"] == 3
    assert sorted(item["value"] for item in op_field["distinct_values"]) == [
        "PREVOD",
        "VKLAD",
        "VYBER",
    ]

    id_field = next(f for f in table["fields"] if f["name"] == "id")
    assert id_field["cardinality"] == 5


def test_json_schema_includes_cardinality_and_distinct_values(tmp_path: Path) -> None:
    json_path = tmp_path / "data.json"
    json_path.write_text(
        json.dumps(
            [
                {"category": "A", "value": 10},
                {"category": "B", "value": 20},
                {"category": "A", "value": 30},
                {"category": "C", "value": 40},
            ]
        ),
        encoding="utf-8",
    )

    from data_agent_baseline.inspectors.semantic_catalog import _read_json_schema

    budget = DataInspectorSampleBudget()
    schema = _read_json_schema(json_path, "data.json", budget)

    cat_field = next(f for f in schema["fields"] if f["name"] == "category")
    assert cat_field["cardinality"] == 3
    assert sorted(item["value"] for item in cat_field["distinct_values"]) == ["A", "B", "C"]

    val_field = next(f for f in schema["fields"] if f["name"] == "value")
    assert val_field["cardinality"] == 4


def test_csv_schema_includes_min_max_for_numeric_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "numeric.csv"
    csv_path.write_text(
        "id,amount,label\n1,100,sale\n2,200,refund\n3,150,sale\n",
        encoding="utf-8",
    )
    from data_agent_baseline.inspectors.semantic_catalog import _read_csv_schema

    budget = DataInspectorSampleBudget()
    schema = _read_csv_schema(csv_path, "numeric.csv", budget)

    amt_field = next(f for f in schema["fields"] if f["name"] == "amount")
    assert amt_field["min_value"] == 100.0
    assert amt_field["max_value"] == 200.0

    id_field = next(f for f in schema["fields"] if f["name"] == "id")
    assert id_field["min_value"] == 1.0
    assert id_field["max_value"] == 3.0

    lbl_field = next(f for f in schema["fields"] if f["name"] == "label")
    assert "min_value" not in lbl_field
    assert "max_value" not in lbl_field


def test_csv_schema_mixed_column_skips_min_max(tmp_path: Path) -> None:
    csv_path = tmp_path / "mixed.csv"
    csv_path.write_text("id,note\n1,hello\n2,world\n", encoding="utf-8")
    from data_agent_baseline.inspectors.semantic_catalog import _read_csv_schema

    budget = DataInspectorSampleBudget()
    schema = _read_csv_schema(csv_path, "mixed.csv", budget)

    note_field = next(f for f in schema["fields"] if f["name"] == "note")
    assert "min_value" not in note_field

    id_field = next(f for f in schema["fields"] if f["name"] == "id")
    assert "min_value" in id_field


def test_sqlite_schema_includes_row_count_and_min_max(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE t (id INTEGER, amt REAL, label TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 100.5, 'sale')")
        conn.execute("INSERT INTO t VALUES (2, 200.0, 'refund')")
        conn.execute("INSERT INTO t VALUES (3, 150.0, 'sale')")

    from data_agent_baseline.inspectors.semantic_catalog import _read_sqlite_schema

    budget = DataInspectorSampleBudget()
    schema = _read_sqlite_schema(db_path, "test.db", budget)

    table = schema["tables"][0]
    assert table["row_count"] == 3

    amt_field = next(f for f in table["fields"] if f["name"] == "amt")
    assert amt_field["min_value"] == 100.5
    assert amt_field["max_value"] == 200.0

    id_field = next(f for f in table["fields"] if f["name"] == "id")
    assert id_field["min_value"] == 1.0
    assert id_field["max_value"] == 3.0

    lbl_field = next(f for f in table["fields"] if f["name"] == "label")
    assert "min_value" not in lbl_field


def test_json_schema_includes_min_max_for_numeric_fields(tmp_path: Path) -> None:
    json_path = tmp_path / "data.json"
    json_path.write_text(
        json.dumps(
            [
                {"score": 95.5, "grade": "A"},
                {"score": 72.0, "grade": "B"},
                {"score": 88.0, "grade": "A"},
            ]
        ),
        encoding="utf-8",
    )
    from data_agent_baseline.inspectors.semantic_catalog import _read_json_schema

    budget = DataInspectorSampleBudget()
    schema = _read_json_schema(json_path, "data.json", budget)

    score_field = next(f for f in schema["fields"] if f["name"] == "score")
    assert score_field["min_value"] == 72.0
    assert score_field["max_value"] == 95.5

    grade_field = next(f for f in schema["fields"] if f["name"] == "grade")
    assert "min_value" not in grade_field


def test_relationship_inference_validates_csv_key_matches_with_data(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_csv_rel"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "users.csv").write_text("id,name\n1,Alice\n2,Bob\n3,Chen\n", encoding="utf-8")
    (context_dir / "orders.csv").write_text(
        "id,user_id,total\n10,1,25\n11,1,30\n12,2,40\n13,3,50\n",
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(task_id="task_csv_rel", difficulty="easy", question="Find order users."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())

    relationship = next(
        rel
        for rel in catalog["relationships"]
        if rel["source"]["asset_path"] == "orders.csv" and rel["source"]["fields"] == ["user_id"]
    )
    assert relationship["target"]["asset_path"] == "users.csv"
    assert relationship["target"]["fields"] == ["id"]
    assert relationship["evidence"]["matched_source_row_ratio"] == pytest.approx(1.0)
    assert relationship["confidence"] >= 0.8


def test_relationship_inference_avoids_low_match_and_type_id_false_positive(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_bad_rel"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "posts.csv").write_text(
        "Id,PostTypeId,Title\n1,1,A\n2,2,B\n3,2,C\n4,9,D\n",
        encoding="utf-8",
    )
    (context_dir / "users.csv").write_text("Id,Name\n100,Alice\n101,Bob\n", encoding="utf-8")
    (context_dir / "events.csv").write_text(
        "Id,UserId\n1,100\n2,999\n3,998\n4,997\n",
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(task_id="task_bad_rel", difficulty="easy", question="Inspect relations."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    pairs = {
        (
            rel["source"]["asset_path"],
            rel["source"]["fields"][0],
            rel["target"]["asset_path"],
            rel["target"]["fields"][0],
        )
        for rel in catalog["relationships"]
    }

    assert ("posts.csv", "PostTypeId", "posts.csv", "Id") not in pairs
    assert ("events.csv", "UserId", "users.csv", "Id") not in pairs


def test_relationship_inference_rejects_row_id_sequence_false_positive(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_row_id_rel"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "patients.csv").write_text(
        "ROW_ID,PatientName\n1,Alice\n2,Bob\n3,Chen\n4,Dina\n",
        encoding="utf-8",
    )
    (context_dir / "admissions.csv").write_text(
        "ROW_ID,AdmissionType\n1,EMERGENCY\n2,ELECTIVE\n3,URGENT\n4,NEWBORN\n",
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(task_id="task_row_id_rel", difficulty="easy", question="Inspect row ids."),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())
    row_id_pairs = [
        rel
        for rel in catalog["relationships"]
        if rel["source"]["fields"] == ["ROW_ID"] or rel["target"]["fields"] == ["ROW_ID"]
    ]

    assert row_id_pairs == []
    assert catalog["derived_views"] == []


def test_relationship_inference_matches_json_field_to_sqlite_key(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_json_sqlite_rel"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "posts.json").write_text(
        json.dumps(
            {
                "records": [
                    {"Id": 10, "OwnerUserId": 1},
                    {"Id": 11, "OwnerUserId": 2},
                    {"Id": 12, "OwnerUserId": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    db_path = context_dir / "users.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE users (Id INTEGER PRIMARY KEY, DisplayName TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        conn.execute("INSERT INTO users VALUES (2, 'Bob')")
    task = PublicTask(
        record=TaskRecord(
            task_id="task_json_sqlite_rel", difficulty="easy", question="Inspect relations."
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())

    relationship = next(
        rel
        for rel in catalog["relationships"]
        if rel["source"]["asset_path"] == "posts.json"
        and rel["source"]["fields"] == ["records.OwnerUserId"]
    )
    assert relationship["target"]["asset_path"] == "users.db"
    assert relationship["target"]["table"] == "users"
    assert relationship["target"]["fields"] == ["Id"]
    assert relationship["evidence"]["matched_source_row_ratio"] == pytest.approx(1.0)


def test_relationship_inference_reports_sqlite_explicit_foreign_key(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_sqlite_fk"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    db_path = context_dir / "shop.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, "
            "FOREIGN KEY(user_id) REFERENCES users(id))"
        )
        conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        conn.execute("INSERT INTO users VALUES (2, 'Bob')")
        conn.execute("INSERT INTO orders VALUES (10, 1)")
        conn.execute("INSERT INTO orders VALUES (11, 2)")
    task = PublicTask(
        record=TaskRecord(
            task_id="task_sqlite_fk", difficulty="easy", question="Inspect relations."
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())

    relationship = next(
        rel
        for rel in catalog["relationships"]
        if rel["source"]["table"] == "orders" and rel["source"]["fields"] == ["user_id"]
    )
    assert relationship["target"]["table"] == "users"
    assert relationship["target"]["fields"] == ["id"]
    assert relationship["evidence"]["explicit_sqlite_foreign_key"] is True
    assert relationship["confidence"] >= 0.95


def test_json_schema_strips_records_prefix_and_preserves_json_path(tmp_path: Path) -> None:
    """For object_with_records JSON, field names must be short (no prefix)
    while json_path retains the full dotted path for internal navigation."""
    json_path_file = tmp_path / "exam.json"
    json_path_file.write_text(
        json.dumps(
            {
                "records": [
                    {"ID": 1, "Thrombosis": 2},
                    {"ID": 2, "Thrombosis": 0},
                ]
            }
        ),
        encoding="utf-8",
    )

    from data_agent_baseline.inspectors.semantic_catalog import _read_json_schema
    from data_agent_baseline.config import DataInspectorSampleBudget

    budget = DataInspectorSampleBudget()
    schema = _read_json_schema(json_path_file, "exam.json", budget)

    assert schema["json_structure"] == "object_with_records"

    id_field = next(f for f in schema["fields"] if f["json_path"] == "records.ID")
    assert id_field["name"] == "ID"
    assert id_field["json_path"] == "records.ID"

    thr_field = next(f for f in schema["fields"] if f["json_path"] == "records.Thrombosis")
    assert thr_field["name"] == "Thrombosis"
    assert thr_field["json_path"] == "records.Thrombosis"

    # Verify short names appear in fields
    field_names = {f["name"] for f in schema["fields"]}
    assert "ID" in field_names
    assert "Thrombosis" in field_names
    assert "records.ID" not in field_names
    assert "records.Thrombosis" not in field_names


def test_data_inspector_config_parses_sample_budget(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "dataset:\n  root_path: data/input\n"
        "agent:\n  model: test\n  api_base: http://localhost/v1\n  api_key: ''\n  api_key_env: TEST\n"
        "  max_steps: 16\n  temperature: 0.0\n  enable_data_inspector: true\n"
        "data_inspector:\n"
        "  sample_budget:\n"
        "    catalog_top_distinct_values: 100\n"
        "    max_doc_tokens: 1250\n"
        "run:\n  output_dir: artifacts/runs\n  max_workers: 4\n  task_timeout_seconds: 600\n",
        encoding="utf-8",
    )
    config = load_app_config(yaml_path)
    assert config.data_inspector.sample_budget.catalog_top_distinct_values == 100
    assert config.data_inspector.sample_budget.max_doc_tokens == 1250


def test_explore_data_globally_returns_json_catalog(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    agent = DataUnderstandingAgent(config=DataInspectorConfig())
    profile, _catalog = agent.explore_data_globally(
        context_dir=task.context_dir,
        task_id=task.task_id,
    )
    assert profile.startswith("{")
    payload = json.loads(profile)
    # Lightweight catalog: no phase/instructions and no full schemas
    assert "task_id" in payload
    assert "query_surfaces" in payload
    assert "structured_tables" not in payload
    assert "documents" in payload
    assert "knowledge_documents" in payload
    assert "phase" not in payload
    assert "schemas" not in payload


def test_receive_problem_injects_catalog_message(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=True,
            data_inspector=DataInspectorConfig(),
        ),
    )
    result = agent.run(task)
    assert result.succeeded is True


def test_master_switch_still_disables_both_nodes(tmp_path: Path) -> None:
    task = _create_task(tmp_path)
    model = ScriptedToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_tool_result",
                        "args": {
                            "tool_name": "execute_python",
                            "tool_args": {
                                "code": "print("
                                + repr(
                                    json.dumps(
                                        {"columns": ["status"], "rows": [["ok"]]},
                                        ensure_ascii=False,
                                    )
                                )
                                + ")",
                            },
                        },
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )
    agent = LangGraphAgent(
        model=model,
        tools=create_default_tool_registry(),
        config=LangGraphAgentConfig(
            max_steps=2,
            enable_data_inspector=False,
            data_inspector=DataInspectorConfig(),
        ),
    )
    result = agent.run(task)
    assert result.succeeded is True
    assert result.global_data_profile is None
    assert result.inspector is None


def test_field_tokens_preserves_cjk_segments() -> None:
    from data_agent_baseline.inspectors.semantic_catalog import _field_tokens, _is_cjk_token

    # Pure CJK field name
    tokens = _field_tokens("公司代码")
    assert "公司代码" in tokens
    assert _is_cjk_token("公司代码")

    # Mixed ASCII + CJK field name
    tokens = _field_tokens("companyCode公司代码")
    assert "company" in tokens
    assert "code" in tokens
    assert "公司代码" in tokens

    # Pure ASCII camelCase field name (no regression)
    tokens = _field_tokens("companyId")
    assert "company" in tokens
    assert "id" in tokens
    assert not any(_is_cjk_token(t) for t in tokens)


def test_relationship_inference_detects_chinese_same_name_keys(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_cn_rel"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    # Two CSVs sharing a Chinese key field "公司代码" with high cardinality
    (context_dir / "dividend.csv").write_text(
        "公司代码,分红金额\n"
        "000001,1.5\n"
        "000002,2.0\n"
        "600000,0.8\n"
        "600036,3.2\n"
        "601318,1.1\n"
        "000858,2.5\n",
        encoding="utf-8",
    )
    (context_dir / "industry.csv").write_text(
        "公司代码,行业名称\n"
        "000001,银行\n"
        "000002,房地产\n"
        "600000,银行\n"
        "600036,保险\n"
        "601318,保险\n"
        "000858,白酒\n",
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(task_id="task_cn_rel", difficulty="easy", question="查看分红与行业"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())

    assert len(catalog["relationships"]) > 0, (
        "Expected relationships for Chinese same-name key fields, got none"
    )
    rel = next(
        rel
        for rel in catalog["relationships"]
        if "公司代码" in rel["source"]["fields"]
        and "公司代码" in rel["target"]["fields"]
    )
    assert rel["relationship_type"] == "lookup_code"
    assert rel["confidence"] >= 0.80


def test_relationship_inference_rejects_chinese_metric_fields(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_cn_metric"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    # Fields with Chinese metric substrings (日期, 名称) should NOT be detected as keys
    (context_dir / "table_a.csv").write_text(
        "序号,截止日期,中文名称,公司代码\n"
        "1,2024-01-01,公司A,000001\n"
        "2,2024-02-01,公司B,000002\n"
        "3,2024-03-01,公司C,600000\n"
        "4,2024-04-01,公司D,600036\n"
        "5,2024-05-01,公司E,601318\n"
        "6,2024-06-01,公司F,000858\n",
        encoding="utf-8",
    )
    (context_dir / "table_b.csv").write_text(
        "序号,截止日期,中文名称,公司代码\n"
        "1,2024-01-15,公司A,000001\n"
        "2,2024-02-15,公司B,000002\n"
        "3,2024-03-15,公司C,600000\n"
        "4,2024-04-15,公司D,600036\n"
        "5,2024-05-15,公司E,601318\n"
        "6,2024-06-15,公司F,000858\n",
        encoding="utf-8",
    )
    task = PublicTask(
        record=TaskRecord(task_id="task_cn_metric", difficulty="easy", question="查看数据"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )

    catalog = build_semantic_catalog(task, budget=DataInspectorSampleBudget())

    # Metric-like Chinese fields should never appear as source keys
    for rel in catalog["relationships"]:
        for field_list in [rel["source"]["fields"], rel["target"]["fields"]]:
            for field in field_list:
                assert field not in ("截止日期", "中文名称", "序号"), (
                    f"Metric field '{field}' was incorrectly detected as a key"
                )
