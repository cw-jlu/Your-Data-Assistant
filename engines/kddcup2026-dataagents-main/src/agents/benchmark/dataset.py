"""公开数据集（`data/public/input/`）的加载器。

通过 `DABenchPublicDataset(root_dir)` 扫描所有 `task_<n>/` 子目录，
把每个任务的 `task.json` 和 `context/` 解析成 `PublicTask`。

关键不变式：
- 目录名 `task_<n>` 中的数字用作排序键（避免字典序把 task_10 排到 task_2 前面）
- `task.json` 里的 `task_id` 必须与目录名完全一致（mismatch 直接报错）
- `context/` 必须存在（Agent 工具完全依赖它）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord

# 约定：所有公开任务目录都以 "task_" 开头，数字部分即任务号
TASK_DIR_PREFIX = "task_"


def task_id_number(task_id: str) -> int:
    """从 `task_<n>` 中提取出数字 n，用作自然数排序键 / range 过滤。

    非法前缀直接抛错：调用方有责任保证仅传入 `task_dirs()` 产出的目录名。
    公共名（无下划线）让 CLI / runner 复用同一份解析逻辑去做 task-range 过滤。
    """
    if not task_id.startswith(TASK_DIR_PREFIX):
        raise ValueError(f"Invalid task id: {task_id}")
    return int(task_id.removeprefix(TASK_DIR_PREFIX))


def _load_task_record(task_json_path: Path) -> TaskRecord:
    """解析单个 `task.json`，校验必填字段。

    必填：{task_id, question}。difficulty 缺失时默认 "unknown"。
    """
    payload = json.loads(task_json_path.read_text(encoding="utf-8"))
    required_keys = {"task_id", "question"}
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise ValueError(
            f"Missing required task.json keys for {task_json_path.parent.name}: "
            f"{sorted(missing_keys)}"
        )

    # 全部显式 str() 转换，防止 YAML/JSON 把数字型 task_id 解析成 int
    return TaskRecord(
        task_id=str(payload["task_id"]),
        difficulty=str(payload.get("difficulty", "unknown")),
        question=str(payload["question"]),
    )


@dataclass(frozen=True, slots=True)
class DABenchPublicDataset:
    """`data/public/input/` 的只读视图。

    该对象是 lazy 的：构造时只保存 `root_dir`，真正的文件扫描发生在
    `task_dirs / get_task / iter_tasks` 等方法被调用时。这样 CLI 打印
    `status` 和实际跑任务可以共用同一个实例而不重复 IO。
    """

    root_dir: Path

    @property
    def exists(self) -> bool:
        """快速判断数据集根目录是否存在（给 `status` 子命令用）。"""
        return self.root_dir.is_dir()

    def task_dirs(self) -> list[Path]:
        """枚举全部任务目录并按任务号升序排列。

        不存在的 root 返回空列表而非抛错，允许 `status` 命令在空环境中优雅降级。
        """
        if not self.exists:
            return []

        task_dirs = [
            path
            for path in self.root_dir.iterdir()
            if path.is_dir() and path.name.startswith(TASK_DIR_PREFIX)
        ]
        # 自然数排序：避免字典序把 task_10 排到 task_2 前面
        task_dirs.sort(key=lambda path: task_id_number(path.name))
        return task_dirs

    def list_task_ids(self) -> list[str]:
        """仅返回任务 ID 列表（如 `["task_1", "task_2", ...]`）。"""
        return [path.name for path in self.task_dirs()]

    def get_task(self, task_id: str) -> PublicTask:
        """按 ID 加载单个任务。

        除读取 `task.json` 外，还校验：
        - `task.json` 存在
        - `task.json` 里的 task_id 与目录名一致
        - `context/` 目录存在
        任一校验失败都抛异常——agent 侧无法处理残缺数据。
        """
        task_dir = self.root_dir / task_id
        task_json_path = task_dir / "task.json"
        if not task_json_path.exists():
            raise FileNotFoundError(f"Missing task.json: {task_json_path}")

        record = _load_task_record(task_json_path)
        # 双向一致性：防止数据打包时文件错位
        if record.task_id != task_dir.name:
            raise ValueError(f"task_id mismatch for {task_dir}: task.json has {record.task_id}")

        context_dir = task_dir / "context"
        if not context_dir.is_dir():
            raise FileNotFoundError(f"Missing context dir: {context_dir}")

        assets = TaskAssets(task_dir=task_dir, context_dir=context_dir)
        return PublicTask(record=record, assets=assets)

    def iter_tasks(self) -> list[PublicTask]:
        """按任务号顺序返回全部任务。需要过滤的调用方在迭代结果上自行筛选。"""
        return [self.get_task(task_dir.name) for task_dir in self.task_dirs()]

    def task_counts(self) -> dict[str, int]:
        """按 difficulty 统计任务数，用于 `status` 子命令的快报。"""
        counts: dict[str, int] = {}
        for task in self.iter_tasks():
            counts[task.difficulty] = counts.get(task.difficulty, 0) + 1
        return counts
