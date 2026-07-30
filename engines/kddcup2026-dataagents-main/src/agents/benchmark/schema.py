"""任务（Task）层面的数据模型定义。

把 `task.json` 里的元信息（`TaskRecord`）与文件系统资产
（`TaskAssets` = task_dir + context_dir）组合成 `PublicTask`，
作为整条流水线（dataset → agent → tools）共享的只读句柄。

`AnswerTable` 是 Agent 最终提交答案的结构化表示——Agent 通过
`answer` 工具返回它，runner 再写成 `prediction.csv`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskRecord:
    """任务元数据，来源于每个任务目录下的 `task.json`。

    - task_id:   形如 `task_1`，与其目录名必须一致
    - difficulty: easy / medium / hard / extreme（见 data/public/README.md）
    - question:  用自然语言描述的问题，是送给 Agent 的核心输入
    """

    task_id: str
    difficulty: str
    question: str


@dataclass(frozen=True, slots=True)
class TaskAssets:
    """任务的文件系统资产定位。

    - task_dir:    `<dataset_root>/task_<id>/`
    - context_dir: `<task_dir>/context/`，Agent 只允许访问该目录下的文件
    """

    task_dir: Path
    context_dir: Path


@dataclass(frozen=True, slots=True)
class PublicTask:
    """组合了元信息和资产的只读任务对象。

    该对象被 dataset 层构造，向下传给 agent 和 tools。通过属性封装让调用方
    不必关心 `record` vs `assets` 的拆分，直接 `task.task_id` / `task.context_dir`。
    """

    record: TaskRecord
    assets: TaskAssets

    # --- 元信息代理属性 ---
    @property
    def task_id(self) -> str:
        return self.record.task_id

    @property
    def difficulty(self) -> str:
        return self.record.difficulty

    @property
    def question(self) -> str:
        return self.record.question

    # --- 路径代理属性 ---
    @property
    def task_dir(self) -> Path:
        return self.assets.task_dir

    @property
    def context_dir(self) -> Path:
        return self.assets.context_dir


@dataclass(frozen=True, slots=True)
class AnswerTable:
    """Agent 提交的答案表格。

    比赛评分按 "无序列值向量" 匹配，因此列名（columns）仅用于对齐输出 CSV
    的表头，但不参与评分；真正决定得分的是各列值的多重集合。
    """

    columns: list[str]
    rows: list[list[Any]]

    def to_dict(self) -> dict[str, Any]:
        """序列化为可直接 JSON 落盘的 dict。

        这里对 columns/rows 都做了浅拷贝，避免外部修改污染 frozen dataclass 内部状态。
        """
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }
