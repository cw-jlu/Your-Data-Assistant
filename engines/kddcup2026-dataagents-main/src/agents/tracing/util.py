"""ID generation and timestamp helpers for the tracing subsystem."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


def gen_trace_id() -> str:
    return uuid.uuid4().hex


def gen_span_id() -> str:
    return uuid.uuid4().hex[:16]


def time_iso() -> str:
    return datetime.now(UTC).isoformat()
