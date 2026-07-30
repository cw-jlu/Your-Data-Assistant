"""Concrete processor and exporter implementations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agents.tracing.processor_interface import TracingExporter, TracingProcessor

if TYPE_CHECKING:
    from agents.tracing.spans import Span
    from agents.tracing.store import SQLiteTraceStore
    from agents.tracing.traces import Trace

logger = logging.getLogger(__name__)


class SQLiteExporter(TracingExporter):
    def __init__(self, store: SQLiteTraceStore) -> None:
        self.store = store

    def export(self, items: list[dict[str, Any] | None]) -> None:
        for item in items:
            if item is None:
                continue
            try:
                if item.get("object") == "trace":
                    self.store.upsert_trace(item)
                elif item.get("object") == "trace.span":
                    self.store.upsert_span(item)
            except Exception:
                logger.exception("SQLiteExporter failed to write item %s", item.get("id"))

    def close(self) -> None:
        self.store.close()


class SimpleProcessor(TracingProcessor):
    def __init__(self, exporter: TracingExporter) -> None:
        self._exporter = exporter

    def on_trace_start(self, trace: Trace) -> None:
        exported = trace.export()
        if exported is not None:
            self._exporter.export([exported])

    def on_trace_end(self, trace: Trace) -> None:
        exported = trace.export()
        if exported is not None:
            self._exporter.export([exported])
            if isinstance(self._exporter, SQLiteExporter):
                self._exporter.store.finalize_trace(exported["id"])

    def on_span_start(self, span: Span[Any]) -> None:
        exported = span.export()
        if exported is not None:
            self._exporter.export([exported])

    def on_span_end(self, span: Span[Any]) -> None:
        exported = span.export()
        if exported is not None:
            self._exporter.export([exported])

    def shutdown(self) -> None:
        if isinstance(self._exporter, SQLiteExporter):
            self._exporter.close()

    def force_flush(self) -> None:
        pass
