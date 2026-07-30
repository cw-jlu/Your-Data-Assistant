"""SQLiteTraceStore — CRUD, finalize, and close behaviour."""

from __future__ import annotations

import threading
from pathlib import Path

from agents.tracing.store import SQLiteTraceStore


def _make_trace(trace_id: str, run_id: str = "r1", task_id: str = "t1") -> dict:
    return {
        "object": "trace",
        "id": trace_id,
        "run_id": run_id,
        "task_id": task_id,
        "name": "TestTrace",
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": None,
        "metadata": {},
    }


def _make_span(
    span_id: str,
    trace_id: str,
    kind: str = "function",
    parent_id: str | None = None,
) -> dict:
    return {
        "object": "trace.span",
        "id": span_id,
        "trace_id": trace_id,
        "parent_id": parent_id,
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:00:01+00:00",
        "span_data": {"type": kind, "name": f"span-{span_id}"},
        "error": None,
    }


def test_upsert_and_list_traces(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    store.upsert_trace(_make_trace("tr1", run_id="r1"))
    store.upsert_trace(_make_trace("tr2", run_id="r1"))

    result = store.list_traces(run_id="r1")
    assert result["total"] == 2
    assert len(result["items"]) == 2
    store.close()


def test_upsert_trace_updates_on_conflict(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    store.upsert_trace(_make_trace("tr1"))
    completed = _make_trace("tr1")
    completed["ended_at"] = "2026-01-01T00:01:00+00:00"
    store.upsert_trace(completed)

    result = store.get_trace_with_spans("tr1")
    assert result is not None
    assert result["trace"]["status"] == "completed"
    store.close()


def test_upsert_trace_with_failure_metadata_sets_error_status(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    trace = _make_trace("tr1")
    trace["ended_at"] = "2026-01-01T00:01:00+00:00"
    trace["metadata"] = {"failure_reason": "Agent did not submit an answer within max_steps."}

    store.upsert_trace(trace)

    result = store.get_trace_with_spans("tr1")
    assert result is not None
    assert result["trace"]["status"] == "error"
    assert result["trace"]["metadata"]["failure_reason"] == (
        "Agent did not submit an answer within max_steps."
    )
    store.close()


def test_upsert_trace_with_timeout_status_stays_timeout(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    trace = _make_trace("tr-timeout")
    trace["ended_at"] = "2026-01-01T00:01:00+00:00"
    trace["metadata"] = {
        "status": "timeout",
        "failure_reason": "Task timed out after 1 seconds.",
    }

    store.upsert_trace(trace)

    result = store.get_trace_with_spans("tr-timeout")
    assert result is not None
    assert result["trace"]["status"] == "timeout"
    assert result["trace"]["metadata"]["failure_reason"] == "Task timed out after 1 seconds."
    store.close()


def test_upsert_span_and_get(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    store.upsert_trace(_make_trace("tr1"))
    store.upsert_span(_make_span("s1", "tr1"))

    span = store.get_span("s1")
    assert span is not None
    assert span["trace_id"] == "tr1"
    assert span["kind"] == "function"
    store.close()


def test_finalize_trace_computes_span_count_and_token_sum(tmp_path: Path) -> None:
    # 非 generation span 计入 span_count 但不贡献 token；generation 的 usage 求和
    store = SQLiteTraceStore(tmp_path / "test.db")
    store.upsert_trace(_make_trace("tr1"))
    store.upsert_span(_make_span("s1", "tr1"))
    gen_span = _make_span("s2", "tr1", kind="generation")
    gen_span["span_data"]["usage"] = {"input_tokens": 100, "output_tokens": 50}
    store.upsert_span(gen_span)
    store.finalize_trace("tr1")

    result = store.get_trace_with_spans("tr1")
    assert result is not None
    assert result["trace"]["span_count"] == 2
    assert result["trace"]["total_tokens"] == 150
    store.close()


def test_mark_running_trace_error_sets_error_and_finalizes(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    store.upsert_trace(_make_trace("tr1", run_id="run-1", task_id="task_1"))
    gen_span = _make_span("s1", "tr1", kind="generation")
    gen_span["span_data"]["usage"] = {"input_tokens": 100, "output_tokens": 50}
    store.upsert_span(gen_span)

    updated = store.mark_running_trace_error(
        run_id="run-1",
        task_id="task_1",
        failure_reason="Task timed out after 1 seconds.",
    )

    assert updated == 1
    result = store.get_trace_with_spans("tr1")
    assert result is not None
    trace = result["trace"]
    assert trace["status"] == "error"
    assert trace["ended_at"] is not None
    assert trace["duration_ms"] is not None
    assert trace["span_count"] == 1
    assert trace["total_tokens"] == 150
    assert trace["metadata"]["failure_reason"] == "Task timed out after 1 seconds."
    store.close()


def test_mark_trace_error_overwrites_completed_failure_trace(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    completed = _make_trace("tr1", run_id="run-1", task_id="task_1")
    completed["ended_at"] = "2026-01-01T00:01:00+00:00"
    store.upsert_trace(completed)

    updated = store.mark_running_trace_error(
        run_id="run-1",
        task_id="task_1",
        failure_reason="Task failed after trace shutdown.",
    )

    assert updated == 1
    result = store.get_trace_with_spans("tr1")
    assert result is not None
    trace = result["trace"]
    assert trace["status"] == "error"
    assert trace["metadata"]["failure_reason"] == "Task failed after trace shutdown."
    store.close()


def test_mark_trace_timeout_overwrites_completed_failure_trace(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    completed = _make_trace("tr1", run_id="run-1", task_id="task_1")
    completed["ended_at"] = "2026-01-01T00:01:00+00:00"
    store.upsert_trace(completed)

    updated = store.mark_running_trace_error(
        run_id="run-1",
        task_id="task_1",
        failure_reason="Task timed out after 1 seconds.",
        status="timeout",
    )

    assert updated == 1
    result = store.get_trace_with_spans("tr1")
    assert result is not None
    trace = result["trace"]
    assert trace["status"] == "timeout"
    assert trace["metadata"]["status"] == "timeout"
    assert trace["metadata"]["failure_reason"] == "Task timed out after 1 seconds."
    store.close()


def test_list_runs(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    store.upsert_trace(_make_trace("tr1", run_id="run-a"))
    store.upsert_trace(_make_trace("tr2", run_id="run-b"))
    store.upsert_trace(_make_trace("tr3", run_id="run-a"))

    runs = store.list_runs()
    by_run = {r["run_id"]: r for r in runs}
    assert by_run["run-a"]["task_count"] == 2
    assert by_run["run-b"]["task_count"] == 1
    store.close()


def test_list_traces_pagination(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    for i in range(5):
        store.upsert_trace(_make_trace(f"tr{i}"))

    page1 = store.list_traces(limit=2, offset=0)
    assert len(page1["items"]) == 2
    assert page1["total"] == 5
    assert page1["has_more"] is True

    page3 = store.list_traces(limit=2, offset=4)
    assert len(page3["items"]) == 1
    assert page3["has_more"] is False
    store.close()


def test_close_prevents_further_queries(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    store.upsert_trace(_make_trace("tr1"))
    store.close()

    import sqlite3

    import pytest

    with pytest.raises(sqlite3.ProgrammingError):
        store.list_runs()


def test_creates_parent_dirs(tmp_path: Path) -> None:
    db_path = tmp_path / "a" / "b" / "c" / "traces.db"
    store = SQLiteTraceStore(db_path)
    store.upsert_trace(_make_trace("tr1"))
    assert db_path.exists()
    store.close()


def test_concurrent_reads_and_writes_share_connection_safely(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    writer_count = 4
    reader_count = 2
    traces_per_writer = 20
    barrier = threading.Barrier(writer_count + reader_count)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def record_error(exc: BaseException) -> None:
        with errors_lock:
            errors.append(exc)

    def writer(writer_idx: int) -> None:
        try:
            barrier.wait(timeout=2)
            for item_idx in range(traces_per_writer):
                trace_id = f"tr-{writer_idx}-{item_idx}"
                store.upsert_trace(
                    _make_trace(trace_id, run_id="run-concurrent", task_id=f"task-{item_idx}")
                )
                store.upsert_span(_make_span(f"sp-{writer_idx}-{item_idx}", trace_id))
        except BaseException as exc:
            record_error(exc)

    def reader() -> None:
        try:
            barrier.wait(timeout=2)
            for _ in range(writer_count * traces_per_writer):
                store.list_runs()
                page = store.list_traces(run_id="run-concurrent", limit=5)
                for item in page["items"]:
                    store.get_trace_with_spans(item["trace_id"])
                store.get_spans_after("2025-01-01T00:00:00+00:00", run_id="run-concurrent")
        except BaseException as exc:
            record_error(exc)

    threads = [
        *(threading.Thread(target=writer, args=(idx,), daemon=True) for idx in range(writer_count)),
        *(threading.Thread(target=reader, daemon=True) for _ in range(reader_count)),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []

    traces = store.list_traces(run_id="run-concurrent", limit=writer_count * traces_per_writer)
    assert traces["total"] == writer_count * traces_per_writer
    assert len(traces["items"]) == writer_count * traces_per_writer
    store.close()
