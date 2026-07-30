"""`read_csv_preview` 的单元测试：锁住 inspect-style 契约。

设计意图回顾（见 `read_csv.read_csv_preview` docstring）：
- read_csv 是样本工具，不给模型"取数据"的错觉。
- 固定返回 `columns + dtypes(approx) + row_count + head(20) + tail(5)`，无 max_rows 旋钮。
- tail 与 head 不重叠：row_count <= 20 时 tail 为空，>20 时只看 head 之后的最后 5 行。
- 任何过滤/聚合任务必须切到 execute_python——本文件不替它把关，只验工具自身行为。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools.read_csv import read_csv_preview


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_csv"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_csv", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_small_file_returns_full_head_and_empty_tail(tmp_path: Path) -> None:
    """row_count <= 20 时 head 覆盖全部数据，tail 应为空（避免重复）。"""
    task = _make_task(tmp_path)
    rows = [[f"id{i}", str(i)] for i in range(10)]
    _write_csv(task.context_dir / "small.csv", ["id", "n"], rows)

    out = read_csv_preview(task, "small.csv")

    assert out["columns"] == ["id", "n"]
    assert out["row_count"] == 10
    assert len(out["head"]) == 10
    assert out["head"] == rows
    assert out["tail"] == []


def test_large_file_returns_head_and_tail_without_overlap(tmp_path: Path) -> None:
    """row_count > 20 时 head=前20、tail=最后5；二者不重叠（task_22 回归）。

    36 行的 income.csv：head 给 0..19，tail 给 31..35——把行 35（task_22 漏掉的
    `2019-09-12` Dues 那条）暴露给模型。
    """
    task = _make_task(tmp_path)
    rows = [[f"row{i}", str(i)] for i in range(36)]
    _write_csv(task.context_dir / "income.csv", ["id", "n"], rows)

    out = read_csv_preview(task, "income.csv")

    assert out["row_count"] == 36
    assert len(out["head"]) == 20
    assert out["head"] == rows[:20]
    assert len(out["tail"]) == 5
    assert out["tail"] == rows[31:36]
    # 锁住"无重叠"不变式：head 末尾索引 19 < tail 起始索引 31
    head_last = out["head"][-1]
    tail_first = out["tail"][0]
    assert head_last == ["row19", "19"]
    assert tail_first == ["row31", "31"]


def test_boundary_row_count_yields_no_overlap_and_partial_tail(tmp_path: Path) -> None:
    """row_count 落在 (head_rows, head_rows + tail_rows] 区间时 tail 应"截短"，

    保证仍然不与 head 重叠：22 行 → head 0..19 已展示前 20 行，tail 只能取 20、21
    （即剩余 2 行），而不是补足 5 行（那样会重新展示行 17-19）。
    """
    task = _make_task(tmp_path)
    rows = [[f"r{i}"] for i in range(22)]
    _write_csv(task.context_dir / "boundary.csv", ["c"], rows)

    out = read_csv_preview(task, "boundary.csv")

    assert out["row_count"] == 22
    assert len(out["head"]) == 20
    assert out["tail"] == [["r20"], ["r21"]]


def test_empty_file_returns_empty_shape(tmp_path: Path) -> None:
    """完全空文件：每个字段都给空集合，避免下游 IndexError。"""
    task = _make_task(tmp_path)
    (task.context_dir / "empty.csv").write_text("")

    out = read_csv_preview(task, "empty.csv")

    assert out == {
        "columns": [],
        "dtypes": [],
        "row_count": 0,
        "head": [],
        "tail": [],
    }


def test_header_only_file_has_zero_row_count(tmp_path: Path) -> None:
    """只有 header、无数据行：dtypes 全 empty，head/tail 都为空。"""
    task = _make_task(tmp_path)
    (task.context_dir / "header_only.csv").write_text("a,b,c\n")

    out = read_csv_preview(task, "header_only.csv")

    assert out["columns"] == ["a", "b", "c"]
    assert out["row_count"] == 0
    assert out["head"] == []
    assert out["tail"] == []
    assert out["dtypes"] == ["empty", "empty", "empty"]


def test_dtype_inference_distinguishes_int_float_str(tmp_path: Path) -> None:
    """dtype 沿用 inspect_files 同套 _classify_dtype：int/float/str 三种基本形。"""
    task = _make_task(tmp_path)
    rows = [
        ["1", "1.5", "alice"],
        ["2", "2.0", "bob"],
        ["3", "3.25", "carol"],
    ]
    _write_csv(task.context_dir / "typed.csv", ["i", "f", "s"], rows)

    out = read_csv_preview(task, "typed.csv")

    assert out["dtypes"] == ["int", "float", "str"]


def test_path_escaping_raises(tmp_path: Path) -> None:
    """安全边界：拒绝走出 context dir。"""
    task = _make_task(tmp_path)
    (task.context_dir / "ok.csv").write_text("a\n1\n")

    with pytest.raises(ValueError, match="escapes context dir"):
        read_csv_preview(task, "../escape.csv")


def test_missing_file_raises(tmp_path: Path) -> None:
    task = _make_task(tmp_path)

    with pytest.raises(FileNotFoundError):
        read_csv_preview(task, "nope.csv")
