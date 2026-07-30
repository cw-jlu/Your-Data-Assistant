"""SQLite 工具运行时行为测试。

重点覆盖 `_maybe_raise_non_sqlite_path_error` 把 sqlite 的
`file is not a database` 翻译成带路由提示的 ValueError——以便 driver
把它打包成下一轮的 `observation.error`，模型一步纠偏。

回归 run 20260430-013 task_16 类故障：模型对 `csv/trans.csv` 调
execute_context_sql，原始观察只有"file is not a database"，下一步往往
继续在 sqlite 工具上挣扎。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from agents.tools._sqlite_common import validate_subquery_columns
from agents.tools.execute_sql import execute_sql
from agents.tools.inspect_sqlite import inspect_sqlite_database

# ---------------------------------------------------------------------------
# Non-SQLite path translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "invoke",
    [
        pytest.param(inspect_sqlite_database, id="inspect_sqlite_schema"),
        pytest.param(
            lambda path: execute_sql(path, "SELECT * FROM sqlite_master"),
            id="execute_sql",
        ),
    ],
)
@pytest.mark.parametrize(
    ("ext", "redirect_keyword"),
    [
        (".csv", "preview_file"),
        (".json", "preview_file"),
        (".md", "preview_file"),
        (".txt", "preview_file"),
    ],
)
def test_sqlite_tools_translate_non_db_path(
    tmp_path: Path, invoke: Callable[[Path], object], ext: str, redirect_keyword: str
) -> None:
    """非 sqlite 路径 → ValueError 含路径名/扩展名/路由提示。"""
    target = tmp_path / f"trans{ext}"
    target.write_text("dummy content\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        invoke(target)

    msg = str(exc_info.value)
    assert target.name in msg
    assert ext in msg
    assert "not a sqlite database" in msg
    assert redirect_keyword in msg
    # 仍要透出"应放弃 sqlite 工具"的元信号
    assert "execute_context_sql" in msg


def test_unknown_extension_falls_back_to_execute_python_hint(tmp_path: Path) -> None:
    """未识别扩展名（如 .parquet）兜底建议 execute_python——不要把模型推回 sqlite 工具。"""
    target = tmp_path / "data.parquet"
    target.write_text("not a real parquet, just bytes for the test", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        inspect_sqlite_database(target)

    msg = str(exc_info.value)
    assert ".parquet" in msg
    assert "execute_python" in msg


def test_no_extension_falls_back_to_execute_python_hint(tmp_path: Path) -> None:
    """无扩展名也要给具体路由（execute_python），不能空回"file is not a database"。"""
    target = tmp_path / "rawdata"
    target.write_text("plain bytes\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        inspect_sqlite_database(target)

    msg = str(exc_info.value)
    assert "extension: none" in msg
    assert "execute_python" in msg


def test_real_sqlite_db_still_works(tmp_path: Path) -> None:
    """正向回归：真 sqlite 文件不被翻译路径误伤。"""
    db_path = tmp_path / "real.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
    conn.commit()
    conn.close()

    schema = inspect_sqlite_database(db_path)
    assert schema["tables"] and schema["tables"][0]["name"] == "t"

    result = execute_sql(db_path, "SELECT id, name FROM t ORDER BY id")
    assert result["columns"] == ["id", "name"]
    assert result["rows"] == [[1, "a"], [2, "b"]]


def test_other_database_errors_pass_through(tmp_path: Path) -> None:
    """非 `file is not a database` 的 DatabaseError 继续向上抛——
    确保翻译逻辑只针对路径类型错误，不吞别的诊断信息。"""
    db_path = tmp_path / "real.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.OperationalError) as exc_info:
        execute_sql(db_path, "SELECT * FROM nonexistent_table")
    assert "no such table" in str(exc_info.value).lower()


def test_empty_result_returns_zero_row_count(tmp_path: Path) -> None:
    """SQL query returning 0 rows has row_count=0 in the result dict."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO items VALUES (1, 'a'), (2, 'b')")
    conn.commit()
    conn.close()

    result = execute_sql(db_path, "SELECT * FROM items WHERE id = 999")
    assert result["row_count"] == 0
    assert result["rows"] == []


# ---------------------------------------------------------------------------
# CREATE INDEX
# ---------------------------------------------------------------------------


def test_create_index(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE orders (id INTEGER, customer TEXT, amount REAL)")
    conn.commit()
    conn.close()

    result = execute_sql(db_path, "CREATE INDEX idx_customer ON orders(customer)")
    assert result["ok"] is True
    assert "created" in result["message"].lower()

    conn = sqlite3.connect(db_path)
    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='orders'"
    ).fetchall()
    conn.close()
    assert any(name == "idx_customer" for (name,) in indexes)


def test_create_index_already_exists_surfaces_sqlite_error(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.execute("CREATE INDEX idx_id ON t(id)")
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        execute_sql(db_path, "CREATE INDEX idx_id ON t(id)")


def test_create_index_name_conflict_on_other_table_surfaces_error(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE a (id INTEGER)")
    conn.execute("CREATE TABLE b (id INTEGER)")
    conn.execute("CREATE INDEX idx_id ON a(id)")
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.OperationalError, match="already exists"):
        execute_sql(db_path, "CREATE INDEX idx_id ON b(id)")


def test_create_index_if_not_exists(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.execute("CREATE INDEX idx_id ON t(id)")
    conn.commit()
    conn.close()

    result = execute_sql(db_path, "CREATE INDEX IF NOT EXISTS idx_id ON t(id)")
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# EXPLAIN QUERY PLAN
# ---------------------------------------------------------------------------


def test_explain_query_plan(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO items VALUES (1, 'a')")
    conn.commit()
    conn.close()

    result = execute_sql(db_path, "EXPLAIN QUERY PLAN SELECT * FROM items WHERE id = 1")
    assert result["ok"] is True
    assert "plan" in result


# ---------------------------------------------------------------------------
# Write operations still blocked
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET id = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "ALTER TABLE t ADD COLUMN x",
    ],
)
def test_write_operations_blocked(tmp_path: Path, sql: str) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="Only SELECT"):
        execute_sql(db_path, sql)


# ---------------------------------------------------------------------------
# Query timeout
# ---------------------------------------------------------------------------


def test_query_timeout_fires(tmp_path: Path) -> None:
    """A query exceeding the timeout raises ValueError with actionable hint."""
    from agents.tools._sqlite_common import _install_query_timeout

    db_path = tmp_path / "slow.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE a (x INTEGER)")
    conn.executemany("INSERT INTO a VALUES (?)", [(i,) for i in range(5000)])
    conn.commit()
    conn.close()

    conn = sqlite3.connect(db_path)
    _install_query_timeout(conn, timeout=0.0)
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        conn.execute(
            "SELECT COUNT(*) FROM a AS t1 CROSS JOIN a AS t2 CROSS JOIN a AS t3"
        ).fetchall()
    conn.close()


def test_timeout_translated_to_actionable_error(tmp_path: Path) -> None:
    """execute_sql translates 'interrupted' into a ValueError with hints."""
    from unittest.mock import patch

    from agents.tools._sqlite_common import _install_query_timeout

    db_path = tmp_path / "slow.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE a (x INTEGER)")
    conn.executemany("INSERT INTO a VALUES (?)", [(i,) for i in range(5000)])
    conn.commit()
    conn.close()

    original_connect = __import__(
        "agents.tools._sqlite_common", fromlist=["connect_read_only"]
    ).connect_read_only

    def connect_with_tiny_timeout(path: Path) -> sqlite3.Connection:
        c = original_connect(path)
        _install_query_timeout(c, timeout=0.0)
        return c

    with (
        patch("agents.tools.execute_sql.connect_read_only", connect_with_tiny_timeout),
        pytest.raises(ValueError, match="timed out"),
    ):
        execute_sql(
            db_path,
            "SELECT COUNT(*) FROM a AS t1 CROSS JOIN a AS t2 CROSS JOIN a AS t3",
        )


def test_explain_timeout_translated_to_actionable_error(tmp_path: Path) -> None:
    """EXPLAIN path translates 'interrupted' into a ValueError with hints."""
    from unittest.mock import MagicMock, patch

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE a (x INTEGER)")
    conn.commit()
    conn.close()

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.OperationalError("interrupted")
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)

    with (
        patch("agents.tools.execute_sql.connect_read_only", return_value=mock_conn),
        pytest.raises(ValueError, match="timed out"),
    ):
        execute_sql(db_path, "EXPLAIN QUERY PLAN SELECT * FROM a")


def test_create_index_timeout_translated_to_actionable_error(tmp_path: Path) -> None:
    """CREATE INDEX path translates 'interrupted' into a ValueError with hints."""
    from unittest.mock import MagicMock, patch

    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE a (x INTEGER)")
    conn.commit()
    conn.close()

    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.OperationalError("interrupted")
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)

    with (
        patch("agents.tools.execute_sql.connect_writable", return_value=mock_conn),
        pytest.raises(ValueError, match="timed out"),
    ):
        execute_sql(db_path, "CREATE INDEX idx_a_x ON a(x)")


# ---------------------------------------------------------------------------
# Subquery column validation
# ---------------------------------------------------------------------------


def test_validate_subquery_columns_warns_on_missing_column() -> None:
    schema = {
        "orders": ["id", "customer_id", "amount"],
        "customers": ["id", "name"],
    }
    sql = "SELECT * FROM orders WHERE customer_id IN (SELECT customer_id FROM customers)"
    warnings = validate_subquery_columns(sql, schema)
    assert len(warnings) == 1
    assert "customer_id" in warnings[0]
    assert "customers" in warnings[0]
    assert "orders" in warnings[0]


def test_validate_subquery_columns_no_warning_when_correct() -> None:
    schema = {
        "orders": ["id", "customer_id", "amount"],
        "customers": ["id", "name"],
    }
    sql = "SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers)"
    warnings = validate_subquery_columns(sql, schema)
    assert warnings == []


def test_validate_subquery_columns_unknown_table_ignored() -> None:
    schema = {"orders": ["id", "amount"]}
    sql = "SELECT * FROM orders WHERE id IN (SELECT id FROM unknown_table)"
    warnings = validate_subquery_columns(sql, schema)
    assert warnings == []


def test_column_validation_blocks_query_before_execution(tmp_path: Path) -> None:
    """Invalid subquery columns fail hard before SQLite returns poisoned rows."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER)")
    conn.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO orders VALUES (1, 10)")
    conn.execute("INSERT INTO orders VALUES (2, 999)")
    conn.execute("INSERT INTO customers VALUES (10, 'Alice')")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError) as exc_info:
        execute_sql(
            db_path,
            "SELECT id FROM orders "
            "WHERE customer_id IN (SELECT customer_id FROM customers) "
            "ORDER BY id",
        )

    msg = str(exc_info.value)
    assert "Unsafe SQL subquery column reference" in msg
    assert "customer_id" in msg
    assert "customers" in msg


# ---------------------------------------------------------------------------
# inspect_sqlite_database row_count
# ---------------------------------------------------------------------------


def test_inspect_includes_row_count(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE big (id INTEGER)")
    for i in range(100):
        conn.execute("INSERT INTO big VALUES (?)", (i,))
    conn.execute("CREATE TABLE empty (id INTEGER)")
    conn.commit()
    conn.close()

    schema = inspect_sqlite_database(db_path)
    tables = {t["name"]: t for t in schema["tables"]}
    assert tables["big"]["row_count"] == 100
    assert tables["empty"]["row_count"] == 0


def test_sqlite_schema_tool_descriptions_include_row_count() -> None:
    from agents.tools.inspect_sqlite import inspect_sqlite_schema
    from agents.tools.preview import preview_file

    for description in (inspect_sqlite_schema.description, preview_file.description):
        assert "{name, create_sql, row_count}" in description
        assert "does not return row counts" not in description.lower()


def test_auto_quote_identifiers_does_not_rewrite_string_literals(tmp_path: Path) -> None:
    """Auto-quoting must not change literal filter values that look like column names."""
    db_path = tmp_path / "quote_retry.db"
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE t (name TEXT, "A/B" INTEGER)')
    conn.execute("INSERT INTO t VALUES (?, ?)", ("A/B", 7))
    conn.commit()
    conn.close()

    result = execute_sql(db_path, "SELECT A/B FROM t WHERE name = 'A/B'")

    assert result["columns"] == ["A/B"]
    assert result["rows"] == [[7]]
    assert result["auto_quoted_sql"] == "SELECT \"A/B\" FROM t WHERE name = 'A/B'"


def test_auto_quote_identifiers_leaves_existing_quoted_identifiers_intact(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "quote_retry.db"
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE t ("A/B" INTEGER, "C-D" INTEGER)')
    conn.execute("INSERT INTO t VALUES (?, ?)", (7, 3))
    conn.commit()
    conn.close()

    result = execute_sql(db_path, 'SELECT A/B, "C-D" FROM t')

    assert result["columns"] == ["A/B", "C-D"]
    assert result["rows"] == [[7, 3]]
    assert result["auto_quoted_sql"] == 'SELECT "A/B", "C-D" FROM t'
