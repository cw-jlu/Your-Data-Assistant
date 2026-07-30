"""路径解析与目录树遍历。"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

from agents.benchmark.schema import PublicTask
from agents.config import ETL_SCRATCH_ROOT


def _link_or_copy_file(source: Path, dest: Path) -> None:
    """Prefer symlinks; copy SQLite files so the agent can CREATE INDEX."""
    import shutil

    from agents.tools.constants import SQLITE_EXTS

    if source.suffix.lower() in SQLITE_EXTS:
        shutil.copy2(source, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IWUSR)
        return
    try:
        dest.symlink_to(source)
    except OSError:
        shutil.copy2(source, dest)


def build_virtual_context(task: PublicTask) -> Path:
    """Create /tmp/dabench/<task_id>/context/ with symlinks merging real context + ETL.

    Prose sources (.md/.pdf) are hidden only when ETL produced a *valid* CSV
    for that stem; files whose ETL failed, was skipped, or produced an
    unreadable CSV stay visible so the agent can fall back to the raw source.
    """
    import shutil

    from agents.etl._identity import validate_csv

    virtual_dir = ETL_SCRATCH_ROOT / task.task_id / "context"
    if virtual_dir.exists():
        shutil.rmtree(virtual_dir)
    virtual_dir.mkdir(parents=True, exist_ok=True)

    real_context = task.context_dir.resolve()
    etl_dir = ETL_SCRATCH_ROOT / task.task_id / "_etl"

    valid_etl_csvs: dict[str, Path] = {}
    if etl_dir.is_dir():
        for etl_file in etl_dir.iterdir():
            if etl_file.suffix == ".csv" and validate_csv(etl_file) is not None:
                valid_etl_csvs[etl_file.stem] = etl_file

    for child in real_context.rglob("*"):
        if not child.is_file():
            continue
        if (
            child.suffix.lower() in {".md", ".pdf"}
            and child.name.lower() != "knowledge.md"
            and child.stem in valid_etl_csvs
        ):
            continue
        rel = child.relative_to(real_context)
        dest = virtual_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            _link_or_copy_file(child, dest)

    if valid_etl_csvs:
        csv_dir = virtual_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        for etl_file in valid_etl_csvs.values():
            dest = csv_dir / etl_file.name
            if not dest.exists():
                _link_or_copy_file(etl_file, dest)

    for dirpath in sorted(virtual_dir.rglob("*"), reverse=True):
        if dirpath.is_dir() and not any(dirpath.iterdir()):
            dirpath.rmdir()

    return virtual_dir


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    """把 Agent 传入的相对路径解析为绝对路径，并防止目录逃逸。

    安全模型：
    - 所有路径必须落在 `task.context_dir` 或 ETL scratch 目录之内
    - 传 `../` 或绝对路径会被规整后触发校验失败
    - 不存在的路径直接抛 FileNotFoundError，避免 Agent 拿到空结果误判

    When context_dir is a virtual context with symlinks, the containment
    check uses ``os.path.normpath`` (no symlink resolution) so files whose
    symlink targets live outside the virtual root are still reachable.
    """
    import os

    normed = Path(os.path.normpath(task.context_dir / relative_path))
    ctx_normed = Path(os.path.normpath(task.context_dir))
    etl_normed = Path(os.path.normpath(ETL_SCRATCH_ROOT / task.task_id))
    if (
        ctx_normed not in normed.parents
        and normed != ctx_normed
        and etl_normed not in normed.parents
        and normed != etl_normed
    ):
        raise ValueError(f"Path escapes context dir: {relative_path}")
    candidate = task.context_dir / relative_path
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate.resolve()


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, Any]:
    """按深度优先遍历 `context/` 下的文件/目录树。

    - `max_depth` 从 1 起算（1 表示只列直接子项）
    - 同目录下 **目录先于文件**（排序键用 `(is_file, name)`），便于模型先看结构再下钻
    - 返回相对路径（POSIX 风格），跨平台一致
    """
    entries: list[dict[str, Any]] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }
