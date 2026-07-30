"""`--range` 选项的解析与 runner 过滤测试。

覆盖：
- `cli._parse_task_range` 各种输入格式（数字 / `task_<n>` / 混合）和错误分支
- `run_benchmark(task_range=...)` 与 `limit` 的组合：先 range 再 limit
- 边界：start > end 在 runner 层也要兜底（防 CLI 之外的调用）
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import typer

from agents.cli import _parse_task_range
from agents.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    RunConfig,
)
from agents.runs import runner as runner_mod
from agents.runs.runner import TaskRunArtifacts

# ---------- _parse_task_range ----------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("5-10", (5, 10)),
        ("task_5-task_10", (5, 10)),
        ("task_5-10", (5, 10)),
        ("5-task_10", (5, 10)),
        ("  5-10  ", (5, 10)),  # 外层 whitespace 也允许
        ("1-1", (1, 1)),  # 单点 range 合法
    ],
)
def test_parse_task_range_happy(spec: str, expected: tuple[int, int]) -> None:
    assert _parse_task_range(spec) == expected


@pytest.mark.parametrize(
    "spec",
    [
        "10",  # 无分隔符
        "",  # 空串
        "-5",  # 左端缺失
        "5-",  # 右端缺失
        "abc-def",  # 非数字
        "task_x-task_y",  # 非数字 task_<n>
        "10-5",  # start > end
        "0-5",  # start < 1
    ],
)
def test_parse_task_range_rejects_bad_input(spec: str) -> None:
    with pytest.raises(typer.BadParameter):
        _parse_task_range(spec)


# ---------- run_benchmark(task_range=...) ----------------------------------------


def _make_config(*, tmp_path: Path) -> AppConfig:
    """构造最小可用 AppConfig；不走 Key 池路径，简化测试断言。"""
    return AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "data"),
        agent=AgentConfig(
            model="m",
            api_base="http://x",
            api_key="primary-key",
        ),
        run=RunConfig(
            output_dir=tmp_path / "runs",
            run_id=None,
            max_workers=1,  # 串行；由 _patch_run_single_task fake 接住调用
            task_timeout_seconds=60,
        ),
    )


class _CapturingSingleTask:
    """记录每次被调到的 task_id，返回占位 artifacts。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(
        self,
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        model: Any = None,
        tools: Any = None,
        answer_verifier: Any = None,
    ) -> TaskRunArtifacts:
        del config, model, tools, answer_verifier
        self.calls.append(task_id)
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            succeeded=True,
            failure_reason=None,
        )


@pytest.mark.parametrize(
    ("task_ids", "task_range", "expected_calls"),
    [
        # 包含 3 / 5 / 8，不含 1 / 2 / 10（注意数据集可以非连续）
        pytest.param(
            ["task_1", "task_2", "task_3", "task_5", "task_8", "task_10"],
            (3, 8),
            ["task_3", "task_5", "task_8"],
            id="filters-non-contiguous-dataset",
        ),
        # range 落在数据集之外：runner 不抛错，只是不跑任何任务（CLI 那层会拦截）
        pytest.param(["task_1", "task_2"], (50, 60), [], id="empty-range-runs-zero-tasks"),
    ],
)
def test_run_benchmark_filters_by_task_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset: Callable[[list[str]], None],
    task_ids: list[str],
    task_range: tuple[int, int],
    expected_calls: list[str],
) -> None:
    fake = _CapturingSingleTask()
    monkeypatch.setattr(runner_mod, "run_single_task", fake)
    patch_dataset(task_ids)

    cfg = _make_config(tmp_path=tmp_path)
    _, artifacts = runner_mod.run_benchmark(config=cfg, task_range=task_range)
    assert fake.calls == expected_calls
    assert [a.task_id for a in artifacts] == expected_calls


def test_run_benchmark_range_then_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset: Callable[[list[str]], None],
) -> None:
    fake = _CapturingSingleTask()
    monkeypatch.setattr(runner_mod, "run_single_task", fake)
    patch_dataset(["task_1", "task_2", "task_3", "task_4", "task_5", "task_6"])

    cfg = _make_config(tmp_path=tmp_path)
    # range 选出 [2..5]，limit=2 再截前两个
    runner_mod.run_benchmark(config=cfg, task_range=(2, 5), limit=2)
    assert fake.calls == ["task_2", "task_3"]


def test_run_benchmark_rejects_inverted_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset: Callable[[list[str]], None],
) -> None:
    """绕过 CLI 直接调 runner 也要兜底校验，防止上游传错 tuple。"""
    fake = _CapturingSingleTask()
    monkeypatch.setattr(runner_mod, "run_single_task", fake)
    patch_dataset(["task_1", "task_2", "task_3"])

    cfg = _make_config(tmp_path=tmp_path)
    with pytest.raises(ValueError, match="task_range start"):
        runner_mod.run_benchmark(config=cfg, task_range=(5, 3))
    # 校验失败时 fake 不应被调用
    assert fake.calls == []


def test_run_benchmark_no_range_runs_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset: Callable[[list[str]], None],
) -> None:
    """task_range=None 时退化回 PR 之前的全量行为。"""
    fake = _CapturingSingleTask()
    monkeypatch.setattr(runner_mod, "run_single_task", fake)
    patch_dataset(["task_1", "task_2", "task_3"])

    cfg = _make_config(tmp_path=tmp_path)
    runner_mod.run_benchmark(config=cfg)
    assert fake.calls == ["task_1", "task_2", "task_3"]
