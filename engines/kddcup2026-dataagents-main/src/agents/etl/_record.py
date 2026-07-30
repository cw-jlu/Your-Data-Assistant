"""Structured intermediate representation for compressed entity data.

压缩阶段之后的管道以 ``pk: v | f: v`` 纯文本为介质时，每个阶段都要自带一套
``split("|") + partition(":")`` 解析，verify/retry 还要用正则替换往文本行里
回填值。本模块把这段管道换成内存表：**解析一次**（:func:`parse_kv_text`），
后续阶段直接操作 :class:`Record`，最后**序列化一次**（:func:`records_to_rows`
+ :func:`to_csv_text`）。

已知边界：KV wire 格式本身没有转义机制，parse 时值内的 ``|`` 仍会截断字段、
值内换行仍会劈断记录行——竖线/换行安全性只对**入库后**（retry/verify 通过
:func:`sanitize_llm_value` 写入 ``Record.fields``）的值成立。
"""

from __future__ import annotations

import contextlib
import csv
import io
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agents.etl._columns import metadata_by_name
from agents.etl._constants import (
    APPROX_TAG,
    ENTITY_LINE,
    RECORD_ID_FIELDS,
    RECORD_ID_LABEL,
    RECORD_ID_RANGE,
    SECTION_HEADING,
)
from agents.etl._merge import (
    canonical_key_map,
    canonicalize_key,
    clean_cell_for_type,
    strip_key,
    strip_reasoning_leak,
)
from agents.etl._types import is_placeholder, split_approx_tag

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# 无 schema 时沿用 ENTITY_LINE 的 "ID:" 约定作为主键键名。
NO_SCHEMA_PK = "ID"

# LLM 值长度上限：字段值是标量，超长内容必然是推理泄漏或原文粘贴。
MAX_LLM_VALUE_CHARS = 500

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_CJK_ASCII_SPACE = re.compile(
    r"([一-鿿㐀-䶿]) +([\x21-\x7e])"
    r"|"
    r"([\x21-\x7e]) +([一-鿿㐀-䶿])"
)


def _new_fields() -> dict[str, str]:
    return {}


def _new_approx() -> set[str]:
    return set()


def _new_provenance() -> dict[str, str]:
    return {}


def _strip_cjk_ascii_spaces(s: str) -> str:
    """Remove spurious spaces between CJK characters and ASCII characters."""
    while _CJK_ASCII_SPACE.search(s):
        s = _CJK_ASCII_SPACE.sub(lambda m: (m[1] or m[3]) + (m[2] or m[4]), s)
    return s


@dataclass(slots=True)
class Reject:
    """A compressed-output line the parser could not turn into a record."""

    line: str
    reason: str  # "no_pk" | "no_kv"


@dataclass(slots=True)
class Record:
    """One entity: canonical-spelling fields (+ unmatched original keys) → bare values."""

    fields: dict[str, str] = field(default_factory=_new_fields)
    pk: str | None = None
    approx: set[str] = field(default_factory=_new_approx)  # 带 ~ 模糊标记的字段名
    provenance: dict[str, str] = field(default_factory=_new_provenance)  # 字段 → 最后写入者
    source_chunk: int | None = None


@dataclass(slots=True)
class EntityTable:
    """In-memory table shared by all post-compression pipeline stages."""

    columns: list[str]
    primary_key: str | None
    anchor_keys: list[str]
    field_types: dict[str, str]
    records: list[Record]
    rejects: list[Reject]
    conflicts: int = 0  # 行内重复键（值不一致）计数

    def stats(self) -> dict[str, int]:
        """Conservation metrics: phases compare before/after to detect silent loss."""
        nonempty = sum(
            1 for record in self.records for val in record.fields.values() if val.strip()
        )
        return {
            "records": len(self.records),
            "nonempty_cells": nonempty,
            "rejects": len(self.rejects),
            "conflicts": self.conflicts,
        }


def _normalize_kv_separators(line: str) -> str:
    """Rewrite ``key=value`` parts to ``key: value`` so parsing sees ``:`` only."""
    if "=" not in line:
        return line
    parts: list[str] = []
    for part in line.split("|"):
        stripped = part.strip()
        equals_at = stripped.find("=")
        colon_at = stripped.find(":")
        if equals_at >= 0 and (colon_at < 0 or equals_at < colon_at):
            key, _, val = stripped.partition("=")
            parts.append(f"{key.strip()}: {val.strip()}")
        else:
            parts.append(stripped)
    return " | ".join(parts)


def parse_kv_text(
    text: str,
    columns: list[str] | None = None,
    primary_key: str | None = None,
    anchor_keys: list[str] | None = None,
    field_types: dict[str, str] | None = None,
) -> EntityTable:
    """Parse compressed KV entity lines into an :class:`EntityTable` — once.

    - 键经 ``canonical_key_map``/``canonicalize_key`` 归一到 schema 拼写；
      未匹配 schema 的键**保留**原名（投影推迟到 :func:`records_to_rows`）。
    - 实体行判定：含规范化主键键名；无 schema 时沿用 ``ID:`` 行首约定。
      缺主键键的 KV 行 → ``Reject("no_pk")``，非 KV 行 → ``Reject("no_kv")``。
      Markdown 结构标题是 trace/section metadata，不参与修复流。
    - 行内重复键：保留首值；后值不一致时计入 ``conflicts``。
    - 值经 ``split_approx_tag`` 剥 ``~`` 入 ``Record.approx``，表内一律存裸值。
    """
    columns = columns or []
    anchor_keys = anchor_keys or []
    key_map = canonical_key_map(columns)
    pk_key = canonicalize_key(primary_key, key_map) if primary_key else NO_SCHEMA_PK
    pk_norm = strip_key(primary_key) if primary_key else None

    records: list[Record] = []
    rejects: list[Reject] = []
    conflicts = 0

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if SECTION_HEADING.match(stripped):
            continue
        normalized = _normalize_kv_separators(stripped)
        if ":" not in normalized:
            rejects.append(Reject(line=stripped, reason="no_kv"))
            continue
        if pk_norm is None and not ENTITY_LINE.match(normalized):
            rejects.append(Reject(line=stripped, reason="no_pk"))
            continue

        record = Record()
        line_conflicts = 0
        has_pk_key = pk_norm is None  # 无 schema 路径：ENTITY_LINE 已匹配
        for part in normalized.split("|"):
            raw_key, sep, val = part.partition(":")
            if not sep:
                continue
            if pk_norm is not None and strip_key(raw_key) == pk_norm:
                has_pk_key = True
            key = canonicalize_key(raw_key, key_map)
            bare, approx = split_approx_tag(val.strip())
            if key in record.fields:
                existing = record.fields[key]
                if existing != bare:
                    line_conflicts += 1
                if is_placeholder(existing) and not is_placeholder(bare):
                    record.fields[key] = bare
                    record.provenance[key] = "parse"
                    if approx and bare:
                        record.approx.add(key)
                    else:
                        record.approx.discard(key)
                continue  # 保留首个有效值
            record.fields[key] = bare
            record.provenance[key] = "parse"
            if approx and bare:
                record.approx.add(key)

        if not has_pk_key:
            rejects.append(Reject(line=stripped, reason="no_pk"))
            continue
        record.pk = record.fields.get(pk_key) or None
        records.append(record)
        conflicts += line_conflicts

    return EntityTable(
        columns=list(columns),
        primary_key=primary_key,
        anchor_keys=list(anchor_keys),
        field_types=dict(field_types or {}),
        records=records,
        rejects=rejects,
        conflicts=conflicts,
    )


def set_record_field(record: Record, col: str, val: str, writer: str) -> None:
    """Store a cleaned cell value on a record, keeping the approx set coherent."""
    bare, approx = split_approx_tag(val)
    record.fields[col] = bare
    if approx and bare:
        record.approx.add(col)
    else:
        record.approx.discard(col)
    record.provenance[col] = writer


def normalize_records(table: EntityTable) -> None:
    """Type-clean every schema-column value in place.

    取代 ``normalize_entity_lines`` 的清洗半边 + 旧 ``deterministic_kv_to_csv``
    的逐格类型清洗（``clean_cell_for_type``：占位符、单位后缀、名称/缩写字段
    数字拦截、fund-type 归一、类型强制）。未匹配 schema 的键不清洗也不丢——
    投影推迟到 :func:`records_to_rows`。
    """
    field_types = metadata_by_name(table.columns, table.field_types)
    col_set = set(table.columns)
    pk = table.primary_key
    for record in table.records:
        for col in record.fields:
            if col not in col_set:
                continue
            raw = record.fields[col]
            if col in record.approx:
                raw = f"{APPROX_TAG}{raw}"
            cleaned = clean_cell_for_type(
                _strip_cjk_ascii_spaces(raw.strip()),
                field_types.get(col),
                field_name=col,
            )
            set_record_field(record, col, cleaned, "normalize")
        if pk:
            record.pk = record.fields.get(pk) or None


def normalize_local_record_ids(table: EntityTable) -> tuple[int, int]:
    """Normalize local record-id labels after parsing and drop range records.

    This is the parser-side replacement for the old compress cleanup that
    converted values like ``档案 286``/``Record 286`` to ``286`` and dropped
    grouped range rows such as ``档案 197至217``. Type cleaning is deliberately
    left to :func:`normalize_records` so field-name reconciliation can run first.

    Returns ``(dropped_records, normalized_values)``.
    """
    normalized = 0
    dropped = 0
    field_types = metadata_by_name(table.columns, table.field_types)
    kept: list[Record] = []
    for record in table.records:
        drop_record = False
        for key, val in list(record.fields.items()):
            field_type = field_types.get(key, "").lower()
            is_local_id = key.lower() in RECORD_ID_FIELDS or "local_record_id" in field_type
            if not is_local_id:
                continue
            if RECORD_ID_RANGE.search(val):
                drop_record = True
                break
            match = RECORD_ID_LABEL.match(val)
            if match:
                set_record_field(record, key, match.group(1), "record_id_normalize")
                normalized += 1
        if drop_record:
            dropped += 1
            continue
        if table.primary_key:
            record.pk = record.fields.get(table.primary_key) or None
        elif "ID" in record.fields:
            record.pk = record.fields.get("ID") or None
        kept.append(record)
    table.records = kept
    return dropped, normalized


def records_to_rows(table: EntityTable) -> tuple[list[str], list[list[str]]]:
    """Project records onto ``table.columns``; re-inject the ``~`` approx prefix.

    回注 ``~`` 让 ``repair_numeric_identities(header, rows)`` 的接口完全不变。
    全空行被丢弃（与旧 deterministic_kv_to_csv 语义一致）。
    """
    header = list(table.columns)
    rows: list[list[str]] = []
    for record in table.records:
        row: list[str] = []
        for col in header:
            val = record.fields.get(col, "")
            if val and col in record.approx:
                val = f"{APPROX_TAG}{val}"
            row.append(val)
        if any(row):
            rows.append(row)
    return header, rows


def table_to_kv_text(table: EntityTable) -> str:
    """Render the table as KV lines for the human-readable trace file only.

    仅供 ``_cache/<stem>_clean.md`` 人读，永不回读进管道。
    """
    lines: list[str] = []
    for record in table.records:
        cols: list[str] = table.columns or list(record.fields)
        parts: list[str] = []
        for col in cols:
            val = record.fields.get(col, "")
            if val and col in record.approx:
                val = f"{APPROX_TAG}{val}"
            parts.append(f"{col}: {val}")
        known: set[str] = set(cols)
        parts.extend(f"{key}: {val}" for key, val in record.fields.items() if key not in known)
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def to_csv_text(header: list[str], rows: list[list[str]]) -> str:
    """Serialize header + rows to CSV text (proper quoting via csv.writer)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue()


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + ``os.replace``).

    读者（agent 侧 validate_csv）永远看不到半写状态。
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def sanitize_llm_value(val: str) -> str:
    """LLM 文本进 ``Record.fields`` 的唯一收口。

    推理泄漏截断 + 控制字符/换行折叠为空格 + 长度上限；落进内存表后，
    值内容不可能再破坏行结构或被当成 regex 模板解释。
    """
    cleaned = strip_reasoning_leak(val)
    cleaned = _CONTROL_CHARS.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned[:MAX_LLM_VALUE_CHARS].rstrip()
