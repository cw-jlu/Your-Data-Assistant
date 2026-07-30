"""Lazy global provider initialization with atexit shutdown."""

from __future__ import annotations

import atexit
import threading

from agents.tracing.provider import DefaultTraceProvider, TraceProvider

_global_trace_provider: TraceProvider | None = None
_lock = threading.Lock()
_shutdown_registered = False


def get_trace_provider() -> TraceProvider:
    global _global_trace_provider, _shutdown_registered
    if _global_trace_provider is not None:
        return _global_trace_provider
    with _lock:
        if _global_trace_provider is not None:
            return _global_trace_provider
        provider = DefaultTraceProvider()
        _global_trace_provider = provider
        if not _shutdown_registered:
            atexit.register(_shutdown_provider)
            _shutdown_registered = True
        return provider


def set_trace_provider(provider: TraceProvider) -> None:
    global _global_trace_provider
    with _lock:
        _global_trace_provider = provider


def _shutdown_provider() -> None:
    if _global_trace_provider is not None:
        _global_trace_provider.shutdown()
