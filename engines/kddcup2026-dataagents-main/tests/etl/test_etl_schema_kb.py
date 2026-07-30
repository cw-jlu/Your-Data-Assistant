from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.etl import _schema_kb


class _FakeCursor:
    def __init__(
        self,
        rows: list[tuple[Any, ...]],
        *,
        description: list[tuple[str]] | None = None,
    ) -> None:
        self._rows = rows
        self.description = description

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def __iter__(self) -> Any:
        return iter(self._rows)


class _FakeConnection:
    def __init__(self, *, has_table: bool = True) -> None:
        self.has_table = has_table
        self.closed = False

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> _FakeCursor:
        del params
        if "ORDER BY name" in sql:
            return _FakeCursor([("first_table",), ("second_table",)])
        if "LOWER(name)=?" in sql:
            return _FakeCursor([("SampleTable",)] if self.has_table else [])
        if sql.startswith("PRAGMA table_info"):
            return _FakeCursor(
                [
                    (0, "id", "INTEGER", 0, None, 0),
                    (1, "amount", "REAL", 0, None, 0),
                    (2, "name", "TEXT", 0, None, 0),
                ]
            )
        if sql.startswith("SELECT *"):
            return _FakeCursor(
                [(1, 2.5, "sample")],
                description=[("id",), ("amount",), ("name",)],
            )
        raise AssertionError(f"unexpected SQL: {sql}")

    def close(self) -> None:
        self.closed = True


def _enable_fake_db(monkeypatch: Any, tmp_path: Path) -> None:
    db_path = tmp_path / "schema_kb.db"
    db_path.touch()
    monkeypatch.setattr(_schema_kb, "_DB_PATH", db_path)


def test_lookup_closes_connection_on_schema_hit(monkeypatch: Any, tmp_path: Path) -> None:
    _enable_fake_db(monkeypatch, tmp_path)
    fake = _FakeConnection()
    monkeypatch.setattr(_schema_kb, "_connect_readonly", lambda: fake)

    result = _schema_kb.lookup("sampletable")

    assert result is not None
    assert result.table_name == "SampleTable"
    assert result.columns == ["id", "amount", "name"]
    assert result.dtypes == {"id": "integer", "amount": "number", "name": "string"}
    assert result.sample_rows == [{"id": 1, "amount": 2.5, "name": "sample"}]
    assert fake.closed


def test_lookup_closes_connection_on_schema_miss(monkeypatch: Any, tmp_path: Path) -> None:
    _enable_fake_db(monkeypatch, tmp_path)
    fake = _FakeConnection(has_table=False)
    monkeypatch.setattr(_schema_kb, "_connect_readonly", lambda: fake)

    assert _schema_kb.lookup("missing") is None
    assert fake.closed


def test_all_table_names_closes_connection(monkeypatch: Any, tmp_path: Path) -> None:
    _enable_fake_db(monkeypatch, tmp_path)
    fake = _FakeConnection()
    monkeypatch.setattr(_schema_kb, "_connect_readonly", lambda: fake)

    assert _schema_kb.all_table_names() == ["first_table", "second_table"]
    assert fake.closed
