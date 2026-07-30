"""tests/runs 共享 fixture。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agents.runs import runner as runner_mod


@pytest.fixture
def patch_dataset(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[str]], None]:
    """把 DABenchPublicDataset 换成产出给定 task_ids 的内存 fake，避免依赖磁盘数据。"""

    def _patch(task_ids: list[str]) -> None:
        class _FakeTask:
            def __init__(self, tid: str) -> None:
                self.task_id = tid

        class _FakeDataset:
            def __init__(self, root: Path) -> None:
                del root

            def iter_tasks(self) -> list[_FakeTask]:
                return [_FakeTask(tid) for tid in task_ids]

        monkeypatch.setattr(runner_mod, "DABenchPublicDataset", _FakeDataset)

    return _patch
