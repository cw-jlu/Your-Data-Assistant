from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord

# 公开数据集中的任务目录统一使用 task_<number> 命名。
TASK_DIR_PREFIX = "task_"


# 从 task_id 中提取数值部分，用于目录排序。
def _task_number(task_id: str) -> int:
    if not task_id.startswith(TASK_DIR_PREFIX):
        raise ValueError(f"Invalid task id: {task_id}")
    return int(task_id.removeprefix(TASK_DIR_PREFIX))


def _task_sort_key(task_id: str) -> tuple[int, int | str]:
    try:
        return (0, _task_number(task_id))
    except ValueError:
        return (1, task_id)


# 从 task.json 中加载任务元信息，并校验 Phase 2 必需字段。
def _load_task_record(task_json_path: Path) -> TaskRecord:
    payload = json.loads(task_json_path.read_text(encoding="utf-8"))
    required_keys = {"task_id", "question"}
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(
            f"Unexpected task.json keys for {task_json_path.parent.name}: "
            f"missing required keys {sorted(missing_keys)}"
        )

    return TaskRecord(
        task_id=str(payload["task_id"]),
        difficulty=str(payload.get("difficulty", "unknown")),
        question=str(payload["question"]),
    )


# 对公开任务数据集的本地目录结构做统一封装。
@dataclass(frozen=True, slots=True)
class DABenchPublicDataset:
    root_dir: Path

    # 数据集根目录是否存在。
    @property
    def exists(self) -> bool:
        return self.root_dir.is_dir()

    # 返回所有任务目录，并按 task 编号排序。
    def task_dirs(self) -> list[Path]:
        if not self.exists:
            return []

        task_dirs = [
            path
            for path in self.root_dir.iterdir()
            if path.is_dir() and path.name.startswith(TASK_DIR_PREFIX)
        ]
        # 优先按数值 task id 排序；如果不是纯数字后缀，则回退到字典序。
        task_dirs.sort(key=lambda path: _task_sort_key(path.name))
        return task_dirs

    # 仅返回任务 ID 列表，适合状态展示等轻量场景。
    def list_task_ids(self) -> list[str]:
        return [path.name for path in self.task_dirs()]

    # 加载单个任务，校验 task.json 和 context/ 是否完整可用。
    def get_task(self, task_id: str) -> PublicTask:
        task_dir = self.root_dir / task_id
        task_json_path = task_dir / "task.json"
        if not task_json_path.exists():
            raise FileNotFoundError(f"Missing task.json: {task_json_path}")

        record = _load_task_record(task_json_path)
        if record.task_id != task_dir.name:
            raise ValueError(f"task_id mismatch for {task_dir}: task.json has {record.task_id}")

        context_dir = task_dir / "context"
        if not context_dir.is_dir():
            raise FileNotFoundError(f"Missing context dir: {context_dir}")

        assets = TaskAssets(task_dir=task_dir, context_dir=context_dir)
        return PublicTask(record=record, assets=assets)

    # 遍历任务，并按 task_id / difficulty 等条件做过滤。
    def iter_tasks(
        self,
        *,
        task_ids: list[str] | None = None,
        difficulty: str | None = None,
        difficulties: list[str] | None = None,
    ) -> list[PublicTask]:
        selected_ids = set(task_ids or [])
        selected_difficulties = set(difficulties or [])
        if difficulty is not None:
            selected_difficulties.add(difficulty)

        tasks: list[PublicTask] = []
        for task_dir in self.task_dirs():
            if selected_ids and task_dir.name not in selected_ids:
                continue
            task = self.get_task(task_dir.name)
            if selected_difficulties and task.difficulty not in selected_difficulties:
                continue
            tasks.append(task)
        return tasks

    # 统计各难度下的任务数量。
    def task_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.iter_tasks():
            counts[task.difficulty] = counts.get(task.difficulty, 0) + 1
        return counts
