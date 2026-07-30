"""validate_csv 必须拒收截断/损坏的缓存 CSV，触发重抽取而非投毒每次运行。"""

from __future__ import annotations

from pathlib import Path

from agents.etl._identity import validate_csv


def test_accepts_well_formed_csv(tmp_path: Path) -> None:
    path = tmp_path / "ok.csv"
    path.write_text("id,name\n1,a\n2,b\n", encoding="utf-8")
    assert validate_csv(path) == (["id", "name"], 2)


def test_rejects_ragged_truncated_tail(tmp_path: Path) -> None:
    path = tmp_path / "ragged.csv"
    path.write_text("id,name,value\n1,a,10\n2,b\n", encoding="utf-8")
    assert validate_csv(path) is None


def test_rejects_all_placeholder_data_rows(tmp_path: Path) -> None:
    path = tmp_path / "placeholder_row.csv"
    path.write_text("id,name,value\n1,a,10\nnull,,null\n", encoding="utf-8")
    assert validate_csv(path) is None


def test_rejects_undecodable_bytes(tmp_path: Path) -> None:
    path = tmp_path / "cut.csv"
    # 多字节 UTF-8 字符（“值” = e5 80 bc）在中间截断
    path.write_bytes("id,value\n1,值".encode()[:-1])
    assert validate_csv(path) is None


def test_rejects_empty_header_column(tmp_path: Path) -> None:
    path = tmp_path / "emptycol.csv"
    path.write_text("id,,value\n1,a,10\n", encoding="utf-8")
    assert validate_csv(path) is None


def test_rejects_missing_and_empty_files(tmp_path: Path) -> None:
    assert validate_csv(tmp_path / "absent.csv") is None
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert validate_csv(empty) is None
    header_only = tmp_path / "header.csv"
    header_only.write_text("id,value\n", encoding="utf-8")
    assert validate_csv(header_only) is None
