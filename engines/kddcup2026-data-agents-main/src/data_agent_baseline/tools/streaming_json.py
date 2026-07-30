"""Streaming JSON tools (H-3 / v7).

Three tools that work on JSON files of any size by parsing them as a token
stream via ``ijson`` rather than loading them whole into memory:

- ``streaming_json_keys(path)``       — discover the schema of a record array
- ``streaming_json_count(path, ...)`` — count records (optionally filtered)
- ``streaming_json_aggregate(path, ...)`` — sum / avg / min / max / count over a numeric field

These exist because v5/v6 forensics showed task_249 (166 MB), task_259 (182 MB),
task_257 (441 MB), task_352, task_396, and task_418 all blow up the standard
``read_json`` (capped) and ``execute_python`` (memory pressure / 30 s timeout).
A streaming parser handles all of them in seconds with O(1) memory.

All three tools take a ``records_path`` (default ``records``) — a top-level
JSON key whose value is the array of records. This matches the public set's
convention of wrapping records inside `{"table": "name", "records": [...]}`.
The records_path also accepts dotted access (e.g. ``payload.items``) for
nested arrays.

Filter / field syntax
---------------------

Both ``streaming_json_count`` and ``streaming_json_aggregate`` accept a
JSON-pointer-ish dotted path inside each record (``user.profile.score``)
and an optional ``filter`` dict. The filter is a flat key→value AND match;
keys are dotted paths, values are exact matches (string equality after
``str()`` cast). Numeric comparisons go through the ``filter_op`` parameter:
``filter_op`` may be ``eq`` (default), ``gt``, ``lt``, ``gte``, ``lte``.

This is intentionally minimal — the agent should fall back to
``execute_python`` for anything more complex.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import ijson  # type: ignore[import-not-found]

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import resolve_context_path


_MAX_KEYS_RETURN = 64
_MAX_KEY_SAMPLES = 5
_DEFAULT_RECORDS_PATH = "records"
_AGGREGATE_OPS: frozenset[str] = frozenset({"count", "sum", "avg", "min", "max"})


def _resolve_path(task: PublicTask, path: str) -> Path | None:
    """Resolve a context-relative path. Returns None on any failure
    (missing file, escape attempt). Caller decides how to surface it.
    """
    try:
        return resolve_context_path(task, path)
    except (FileNotFoundError, ValueError):
        return None


def _records_prefix(records_path: str) -> str:
    """Convert a dotted records path to an ijson prefix.

    ``records`` → ``records.item``
    ``payload.items`` → ``payload.items.item``
    """
    cleaned = records_path.strip().strip(".")
    if not cleaned:
        cleaned = _DEFAULT_RECORDS_PATH
    return f"{cleaned}.item"


def _extract_dotted(record: Any, dotted: str) -> Any:
    """Descend a dotted path inside a record. Missing keys → None."""
    if not dotted:
        return record
    parts = dotted.split(".")
    cursor = record
    for part in parts:
        if isinstance(cursor, dict):
            cursor = cursor.get(part)
        elif isinstance(cursor, list):
            try:
                cursor = cursor[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if cursor is None:
            return None
    return cursor


def _matches_filter(record: Any, filt: dict[str, Any] | None, filter_op: str = "eq") -> bool:
    if not filt:
        return True
    op = (filter_op or "eq").lower()
    for dotted, expected in filt.items():
        actual = _extract_dotted(record, dotted)
        if op == "eq":
            if str(actual) != str(expected):
                return False
        elif op in ("gt", "lt", "gte", "lte"):
            try:
                a = float(actual)  # type: ignore[arg-type]
                e = float(expected)
            except (TypeError, ValueError):
                return False
            if op == "gt" and not (a > e):
                return False
            if op == "lt" and not (a < e):
                return False
            if op == "gte" and not (a >= e):
                return False
            if op == "lte" and not (a <= e):
                return False
        else:
            # Unknown op — treat as eq for safety.
            if str(actual) != str(expected):
                return False
    return True


def _iter_records(file_path: Path, records_path: str) -> Iterator[Any]:
    prefix = _records_prefix(records_path)
    with file_path.open("rb") as handle:
        yield from ijson.items(handle, prefix)


def streaming_json_keys(
    task: PublicTask,
    path: str,
    *,
    records_path: str = _DEFAULT_RECORDS_PATH,
    sample_records: int = 50,
) -> dict[str, Any]:
    """Sample the first ``sample_records`` records and report their key shape.

    Returns ``{"path": ..., "records_path": ..., "sample_size": N,
              "keys": [{"key": str, "samples": [str, ...], "non_null_count": int}, ...],
              "total_records_scanned_for_keys": N}``.
    """
    file_path = _resolve_path(task, path)
    if file_path is None or not file_path.is_file():
        return {"path": path, "error": f"file not found: {path}"}

    sample_records = max(1, min(int(sample_records), 500))
    keys_seen: dict[str, list[str]] = {}
    non_null: dict[str, int] = {}
    scanned = 0

    try:
        for record in _iter_records(file_path, records_path):
            scanned += 1
            if isinstance(record, dict):
                for k, v in record.items():
                    samples = keys_seen.setdefault(k, [])
                    if len(samples) < _MAX_KEY_SAMPLES and v is not None:
                        samples.append(str(v)[:80])
                    if v is not None:
                        non_null[k] = non_null.get(k, 0) + 1
            if scanned >= sample_records:
                break
    except Exception as exc:  # noqa: BLE001
        return {
            "path": path,
            "records_path": records_path,
            "error": f"streaming parse failed: {exc}",
            "scanned": scanned,
        }

    keys_summary = [
        {
            "key": k,
            "samples": keys_seen[k],
            "non_null_count": non_null.get(k, 0),
        }
        for k in list(keys_seen.keys())[:_MAX_KEYS_RETURN]
    ]
    return {
        "path": path,
        "records_path": records_path,
        "sample_size": scanned,
        "keys": keys_summary,
        "note": "Sampled the first {} records; counts/samples reflect that subset only.".format(scanned),
    }


def streaming_json_count(
    task: PublicTask,
    path: str,
    *,
    records_path: str = _DEFAULT_RECORDS_PATH,
    filter: dict[str, Any] | None = None,
    filter_op: str = "eq",
) -> dict[str, Any]:
    """Count records (optionally matching ``filter``).

    Returns ``{"path": ..., "total": int, "matched": int}``.
    Without a filter, ``matched == total``.
    """
    file_path = _resolve_path(task, path)
    if file_path is None or not file_path.is_file():
        return {"path": path, "error": f"file not found: {path}"}

    total = 0
    matched = 0
    try:
        for record in _iter_records(file_path, records_path):
            total += 1
            if _matches_filter(record, filter, filter_op):
                matched += 1
    except Exception as exc:  # noqa: BLE001
        return {
            "path": path,
            "records_path": records_path,
            "error": f"streaming parse failed: {exc}",
            "total_so_far": total,
        }

    return {
        "path": path,
        "records_path": records_path,
        "total": total,
        "matched": matched,
        "filter": filter or {},
        "filter_op": filter_op,
    }


def streaming_json_aggregate(
    task: PublicTask,
    path: str,
    *,
    field: str,
    operation: str,
    records_path: str = _DEFAULT_RECORDS_PATH,
    filter: dict[str, Any] | None = None,
    filter_op: str = "eq",
) -> dict[str, Any]:
    """Compute ``operation`` over ``field`` (dotted path inside each record).

    ``operation`` ∈ {count, sum, avg, min, max}. Numeric fields only
    (non-numeric values are skipped, counted as ``skipped``).

    Returns ``{"path": ..., "field": ..., "operation": ..., "result": float | int,
              "n": int, "skipped": int, "matched_records": int}``.
    """
    op = (operation or "").strip().lower()
    if op not in _AGGREGATE_OPS:
        return {
            "path": path,
            "error": f"unsupported operation '{operation}'; choose from {sorted(_AGGREGATE_OPS)}",
        }

    file_path = _resolve_path(task, path)
    if file_path is None or not file_path.is_file():
        return {"path": path, "error": f"file not found: {path}"}

    matched = 0
    n_numeric = 0
    skipped = 0
    sum_val = 0.0
    min_val: float | None = None
    max_val: float | None = None

    try:
        for record in _iter_records(file_path, records_path):
            if not _matches_filter(record, filter, filter_op):
                continue
            matched += 1
            if op == "count":
                continue
            value = _extract_dotted(record, field)
            try:
                fval = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                skipped += 1
                continue
            n_numeric += 1
            sum_val += fval
            if min_val is None or fval < min_val:
                min_val = fval
            if max_val is None or fval > max_val:
                max_val = fval
    except Exception as exc:  # noqa: BLE001
        return {
            "path": path,
            "records_path": records_path,
            "error": f"streaming parse failed: {exc}",
            "matched_so_far": matched,
        }

    result: float | int | None
    if op == "count":
        result = matched
    elif op == "sum":
        result = sum_val
    elif op == "avg":
        result = (sum_val / n_numeric) if n_numeric else None
    elif op == "min":
        result = min_val
    elif op == "max":
        result = max_val
    else:  # unreachable
        result = None

    return {
        "path": path,
        "records_path": records_path,
        "field": field,
        "operation": op,
        "filter": filter or {},
        "filter_op": filter_op,
        "matched_records": matched,
        "n": n_numeric,
        "skipped": skipped,
        "result": result,
    }


__all__ = [
    "streaming_json_keys",
    "streaming_json_count",
    "streaming_json_aggregate",
]
