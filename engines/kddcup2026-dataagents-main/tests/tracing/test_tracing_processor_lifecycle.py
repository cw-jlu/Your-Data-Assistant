"""Processor registration/unregistration and scoped shutdown."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.tracing.processor_interface import TracingProcessor
from agents.tracing.processors import SimpleProcessor, SQLiteExporter
from agents.tracing.provider import (
    DefaultTraceProvider,
    SynchronousMultiTracingProcessor,
)
from agents.tracing.store import SQLiteTraceStore


class _RecordingProcessor(TracingProcessor):
    def __init__(self) -> None:
        self.events: list[str] = []
        self.shut_down = False

    def on_trace_start(self, trace: Any) -> None:
        self.events.append("trace_start")

    def on_trace_end(self, trace: Any) -> None:
        self.events.append("trace_end")

    def on_span_start(self, span: Any) -> None:
        self.events.append("span_start")

    def on_span_end(self, span: Any) -> None:
        self.events.append("span_end")

    def shutdown(self) -> None:
        self.shut_down = True

    def force_flush(self) -> None:
        pass


def test_multi_processor_add_and_remove() -> None:
    mp = SynchronousMultiTracingProcessor()
    p1 = _RecordingProcessor()
    p2 = _RecordingProcessor()
    mp.add_processor(p1)
    mp.add_processor(p2)

    mp.on_trace_start(None)  # type: ignore[arg-type]
    assert p1.events == ["trace_start"]
    assert p2.events == ["trace_start"]

    mp.remove_processor(p1)
    mp.remove_processor(p1)  # removing an unregistered processor must be a no-op
    mp.on_trace_end(None)  # type: ignore[arg-type]
    assert "trace_end" not in p1.events
    assert p2.events == ["trace_start", "trace_end"]


def test_provider_register_and_unregister() -> None:
    provider = DefaultTraceProvider()
    p = _RecordingProcessor()
    provider.register_processor(p)

    t = provider.create_trace(name="test", trace_id="t1", run_id="r1", task_id="tk1")
    t.start()
    assert p.events == ["trace_start"]

    provider.unregister_processor(p)
    t.finish()
    assert "trace_end" not in p.events


def test_shutdown_does_not_affect_unregistered_processor() -> None:
    provider = DefaultTraceProvider()
    p1 = _RecordingProcessor()
    p2 = _RecordingProcessor()
    provider.register_processor(p1)
    provider.register_processor(p2)

    provider.unregister_processor(p1)
    provider.shutdown()
    assert not p1.shut_down
    assert p2.shut_down


def test_simple_processor_shutdown_closes_sqlite_store(tmp_path: Path) -> None:
    store = SQLiteTraceStore(tmp_path / "test.db")
    exporter = SQLiteExporter(store)
    processor = SimpleProcessor(exporter)

    processor.shutdown()

    import sqlite3

    import pytest

    with pytest.raises(sqlite3.ProgrammingError):
        store.list_runs()


def test_scoped_processor_lifecycle(tmp_path: Path) -> None:
    """Simulate the fixed run_single_task_core pattern: register → use → unregister → shutdown."""
    provider = DefaultTraceProvider()
    store = SQLiteTraceStore(tmp_path / "test.db")
    exporter = SQLiteExporter(store)
    processor = SimpleProcessor(exporter)
    provider.register_processor(processor)

    t = provider.create_trace(name="task1", trace_id="t1", run_id="r1", task_id="tk1")
    t.start()
    t.finish()

    traces = store.list_traces()
    assert traces["total"] == 1

    provider.unregister_processor(processor)
    processor.shutdown()

    other = _RecordingProcessor()
    provider.register_processor(other)
    t2 = provider.create_trace(name="task2", trace_id="t2", run_id="r1", task_id="tk2")
    t2.start()
    assert other.events == ["trace_start"]
    t2.finish()
    provider.unregister_processor(other)


def test_disabled_provider_creates_noop_traces() -> None:
    provider = DefaultTraceProvider()
    provider.set_disabled(True)

    t = provider.create_trace(name="test")
    assert t.trace_id == "no-op"
    assert t.export() is None
