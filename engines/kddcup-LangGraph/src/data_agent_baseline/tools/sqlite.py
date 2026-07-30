from __future__ import annotations

import sqlite3
from pathlib import Path


# 以只读模式连接 sqlite 文件，避免工具对原始数据做修改。
def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


# 执行只读 SQL，并限制返回行数，避免一次性返回过大结果。
def execute_read_only_sql(path: Path, sql: str, *, limit: int | None = 200) -> dict[str, object]:
    normalized_sql = sql.lstrip().lower()
    if not normalized_sql.startswith(("select", "with", "pragma")):
        raise ValueError("Only read-only SQL statements are allowed.")

    conn = _connect_read_only(path)
    try:
        cursor = conn.execute(sql)
        column_names = [item[0] for item in cursor.description or []]
        if limit is None:
            rows = cursor.fetchall()
        else:
            rows = cursor.fetchmany(limit + 1)
    finally:
        conn.close()

    if limit is None:
        truncated = False
        limited_rows = rows
    else:
        # 多取一行用于判断是否被截断，但真正返回时只保留 limit 行。
        truncated = len(rows) > limit
        limited_rows = rows[:limit]
    return {
        "path": str(path),
        "columns": column_names,
        "rows": [list(row) for row in limited_rows],
        "row_count": len(limited_rows),
        "truncated": truncated,
    }
