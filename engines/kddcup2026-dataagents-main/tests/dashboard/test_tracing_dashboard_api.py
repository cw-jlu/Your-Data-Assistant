"""Dashboard REST API — endpoint integration tests using Starlette TestClient."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.tracing.store import SQLiteTraceStore


def test_stream_timestamp_advances_to_latest_returned_span_timestamp() -> None:
    from agents.dashboard.api import _advance_stream_timestamp

    last_ts = "2026-01-01T00:00:00+00:00"
    spans = [
        {"started_at": "2026-01-01T00:00:05+00:00", "ended_at": None},
        {
            "started_at": "2026-01-01T00:00:03+00:00",
            "ended_at": "2026-01-01T00:00:09+00:00",
        },
        {"started_at": None, "ended_at": "2026-01-01T00:00:07+00:00"},
    ]

    assert _advance_stream_timestamp(last_ts, spans) == "2026-01-01T00:00:09+00:00"


def test_stream_timestamp_stays_put_when_no_spans_returned() -> None:
    from agents.dashboard.api import _advance_stream_timestamp

    last_ts = "2026-01-01T00:00:00+00:00"

    assert _advance_stream_timestamp(last_ts, []) == last_ts


@pytest.fixture()
def store_and_client(tmp_path: Path):
    store = SQLiteTraceStore(tmp_path / "test.db")
    store.upsert_trace(
        {
            "object": "trace",
            "id": "tr1",
            "run_id": "run1",
            "task_id": "task1",
            "name": "TestTrace",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:01:00+00:00",
            "metadata": {"foo": "bar"},
        }
    )
    store.upsert_span(
        {
            "object": "trace.span",
            "id": "sp1",
            "trace_id": "tr1",
            "parent_id": None,
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:00:30+00:00",
            "span_data": {"type": "function", "name": "test_func"},
            "error": None,
        }
    )

    from fastapi import FastAPI
    from starlette.testclient import TestClient

    app = FastAPI()
    from agents.dashboard.api import create_api_routes

    create_api_routes(app, store)

    yield store, TestClient(app)
    store.close()


def test_list_runs(store_and_client: tuple) -> None:
    _, client = store_and_client
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["run_id"] == "run1"


def test_list_traces(store_and_client: tuple) -> None:
    _, client = store_and_client
    resp = client.get("/api/traces")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["trace_id"] == "tr1"


def test_list_traces_filter_by_run(store_and_client: tuple) -> None:
    _, client = store_and_client
    resp = client.get("/api/traces?run_id=nonexistent")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


def test_get_trace(store_and_client: tuple) -> None:
    _, client = store_and_client
    resp = client.get("/api/traces/tr1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["trace"]["trace_id"] == "tr1"
    assert len(data["spans"]) == 1


def test_get_span(store_and_client: tuple) -> None:
    _, client = store_and_client
    resp = client.get("/api/spans/sp1")
    assert resp.status_code == 200
    assert resp.json()["span_id"] == "sp1"


@pytest.mark.parametrize("path", ["/api/traces/nonexistent", "/api/spans/nonexistent"])
def test_get_missing_resource_returns_404(store_and_client: tuple, path: str) -> None:
    _, client = store_and_client
    assert client.get(path).status_code == 404


def test_dashboard_root_returns_404_when_static_index_is_missing(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from agents.dashboard.app import create_app

    app = create_app(tmp_path / "traces.db", static_dir=tmp_path / "missing-static")

    resp = TestClient(app).get("/")

    assert resp.status_code == 404
    assert "Dashboard static index not found" in resp.json()["detail"]
