"""`create_run_id` daily-counter 命名规则测试。

run_id 格式：`<UTC YYYYMMDD>-<NNN>`，例如 `20260430-001`。
- 日期前缀按 UTC 取，避免本地时区漂移影响排序
- 3 位计数器零填充：`001..999`，超出抛 RuntimeError
- 同进程内多次调用必须配合 `create_run_output_dir` 才能拿到不同 id（计数依赖磁盘扫描）
- 旧 22 字符微秒格式（无 `-` 分隔符）不会与新格式互相干扰，老 run 目录原样保留
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agents.runs.runner import create_run_id, create_run_output_dir

_RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{3}$")


def test_create_run_id_format_is_date_dash_counter(tmp_path: Path) -> None:
    """空目录下应返回 `<today>-001`，整体匹配 `YYYYMMDD-NNN`。"""
    rid = create_run_id(tmp_path)
    assert _RUN_ID_PATTERN.match(rid), f"unexpected format: {rid!r}"
    today = datetime.now(UTC).strftime("%Y%m%d")
    assert rid == f"{today}-001"


def test_create_run_id_starts_at_001_when_dir_missing(tmp_path: Path) -> None:
    """output_root 不存在时不应炸——返回 001（mkdir 会在后续步骤建出来）。"""
    missing = tmp_path / "not-yet-created"
    rid = create_run_id(missing)
    today = datetime.now(UTC).strftime("%Y%m%d")
    assert rid == f"{today}-001"


def test_create_run_id_increments_after_each_dir_creation(tmp_path: Path) -> None:
    """连续 5 次 `create_run_output_dir` 应得到 001..005——靠磁盘上的目录"打卡"。"""
    today = datetime.now(UTC).strftime("%Y%m%d")
    ids: list[str] = []
    for _ in range(5):
        rid, _ = create_run_output_dir(tmp_path)
        ids.append(rid)
    assert ids == [f"{today}-{i:03d}" for i in range(1, 6)]


def test_create_run_id_ignores_unrelated_entries(tmp_path: Path) -> None:
    """非 `<today>-NNN` 命名的目录不应参与计数：
    - 旧微秒格式目录（无 `-` 或长度不符）
    - 其他日期前缀
    - 文件而非目录
    - 后缀含非数字字符
    """
    today = datetime.now(UTC).strftime("%Y%m%d")
    (tmp_path / "20260420T123045123456Z").mkdir()  # 旧格式
    (tmp_path / "20250101-001").mkdir()  # 不同日期
    (tmp_path / f"{today}-abc").mkdir()  # 后缀非数字
    (tmp_path / f"{today}-1234").mkdir()  # 后缀长度不符
    (tmp_path / f"{today}-001").write_text("regular file, not a dir")  # 文件
    rid = create_run_id(tmp_path)
    assert rid == f"{today}-001"


def test_create_run_id_resumes_from_max_existing(tmp_path: Path) -> None:
    """已有 001/002/005 三个目录时（不连续），下一个应是 006（max+1）。"""
    today = datetime.now(UTC).strftime("%Y%m%d")
    for n in (1, 2, 5):
        (tmp_path / f"{today}-{n:03d}").mkdir()
    rid = create_run_id(tmp_path)
    assert rid == f"{today}-006"


def test_create_run_id_raises_when_counter_exhausted(tmp_path: Path) -> None:
    """单日 999 上限：模拟 999 已存在时第 1000 次必须显式抛错而不是悄悄返回 1000。

    用一个空文件占位 999 标识更轻量；此处简化：直接建一个 999 目录代表"今日已用满"。
    """
    today = datetime.now(UTC).strftime("%Y%m%d")
    (tmp_path / f"{today}-999").mkdir()
    with pytest.raises(RuntimeError, match="exceeded 999"):
        create_run_id(tmp_path)


def test_create_run_output_dir_handles_concurrent_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """并发场景：两个 worker 同时算到 001，mkdir 互斥失败一方应自动重试拿到 002。

    模拟方式：第一次 mkdir 前预先建好 `<today>-001`（模拟另一个 worker 抢先成功），
    `create_run_output_dir(run_id=None)` 应不报错，而是返回 002。
    """
    today = datetime.now(UTC).strftime("%Y%m%d")
    (tmp_path / f"{today}-001").mkdir()  # 抢占成功的 worker 留下的痕迹

    rid, run_dir = create_run_output_dir(tmp_path)
    assert rid == f"{today}-002"
    assert run_dir.is_dir()


def test_create_run_output_dir_explicit_run_id_unchanged(tmp_path: Path) -> None:
    """显式 run_id 不走计数器：保持原行为——单段目录名 + 撞名报错。"""
    rid, run_dir = create_run_output_dir(tmp_path, run_id="my-debug-run")
    assert rid == "my-debug-run"
    assert run_dir == tmp_path / "my-debug-run"
    # 撞名应抛 FileExistsError（保留历史防覆盖语义）
    with pytest.raises(FileExistsError):
        create_run_output_dir(tmp_path, run_id="my-debug-run")
