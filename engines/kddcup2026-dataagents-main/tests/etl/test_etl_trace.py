"""Tests for ETL phase tracing — _trace.py."""

from __future__ import annotations

import time

import pytest

from agents.etl._trace import (
    ETLFileTrace,
    etl_phase,
    reset_etl_trace,
    serialize_trace,
    set_etl_trace,
)


@pytest.fixture()
def file_trace():
    t = ETLFileTrace(task_id="task_99", stem="test_doc")
    tokens = set_etl_trace(t)
    yield t
    reset_etl_trace(tokens)


def test_etl_phase_records_timing(file_trace: ETLFileTrace) -> None:
    with etl_phase("compress") as p:
        time.sleep(0.01)
    assert p.started_at > 0
    assert p.ended_at >= p.started_at
    assert len(file_trace.phases) == 1
    assert file_trace.phases[0].name == "compress"


def test_etl_phase_records_error(file_trace: ETLFileTrace) -> None:
    with pytest.raises(ValueError, match="boom"), etl_phase("verify") as p:
        raise ValueError("boom")
    assert p.error is not None
    assert "boom" in p.error
    assert len(file_trace.phases) == 1


def test_etl_phase_metrics(file_trace: ETLFileTrace) -> None:
    with etl_phase("csv", input_size=5000) as p:
        p.metrics["rows"] = 42
        p.decisions["format"] = "pipe-separated"
    assert p.metrics["rows"] == 42
    assert p.metrics["input_size"] == 5000
    assert p.decisions["format"] == "pipe-separated"


def test_etl_file_trace_serialization(file_trace: ETLFileTrace) -> None:
    with etl_phase("schema_parse", input_size=1000) as p:
        p.output_size = 18
        p.metrics["column_count"] = 18
    with etl_phase("compress", input_size=50000) as p:
        p.output_size = 10000
        p.llm_calls = 4

    data = serialize_trace(file_trace)
    assert data["version"] == 1
    assert data["task_id"] == "task_99"
    assert data["stem"] == "test_doc"
    assert isinstance(data["duration_ms"], int)
    assert data["total_llm_calls"] == 4
    assert len(data["phases"]) == 2
    assert data["phases"][0]["name"] == "schema_parse"
    assert data["phases"][1]["name"] == "compress"
    assert data["phases"][1]["llm_calls"] == 4


def test_etl_trace_noop_when_none() -> None:
    """etl_phase works when ContextVar is None (tracing disabled)."""
    with etl_phase("compress") as p:
        p.metrics["chars"] = 100
    assert p.metrics["chars"] == 100
    assert p.started_at > 0
    assert p.ended_at >= p.started_at
