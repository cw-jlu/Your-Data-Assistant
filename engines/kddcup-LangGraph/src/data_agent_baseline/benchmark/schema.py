from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


# task.json 中的结构化元信息。
@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    difficulty: str
    question: str


# 一个任务在文件系统上的资产位置。
@dataclass(frozen=True, slots=True)
class ContextAsset:
    visible_path: str
    physical_path: Path
    source_path: str | None = None
    action: str = "source"
    generated: bool = False


@dataclass(frozen=True, slots=True)
class ContextView:
    source_context_dir: Path
    generated_context_dir: Path
    assets: tuple[ContextAsset, ...]


@dataclass(frozen=True, slots=True)
class TaskAssets:
    task_dir: Path
    context_dir: Path
    context_view: ContextView | None = None


# 运行时看到的完整公开任务对象，由记录信息和文件资产组成。
@dataclass(frozen=True, slots=True)
class PublicTask:
    record: TaskRecord
    assets: TaskAssets

    # 以下属性只是把内部 record / assets 上的字段直接透出，便于调用方使用。
    @property
    def task_id(self) -> str:
        return self.record.task_id

    @property
    def difficulty(self) -> str:
        return self.record.difficulty

    @property
    def question(self) -> str:
        return self.record.question

    @property
    def task_dir(self) -> Path:
        return self.assets.task_dir

    @property
    def context_dir(self) -> Path:
        return self.assets.context_dir


# 最终答案统一表示成表格结构，便于写出为 prediction.csv。
@dataclass(frozen=True, slots=True)
class AnswerTable:
    columns: list[str]
    rows: list[list[Any]]

    # 转成普通字典，供 trace.json 序列化使用。
    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }
