"""`run.blocklist` 跳过名单端到端测试。

覆盖三层：
- Pydantic 校验拒收（负数 / 重复 / 类型错误；"task_<n>" 归一化经由重复判定间接覆盖）
- `run_single_task` 短路：命中名单时不走 timeout 子进程，合成失败 payload
- `run_benchmark` 整合：blocklist + 限制选项叠加时统计与 summary.json 形态正确
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agents.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    RunConfig,
    load_app_config,
)
from agents.runs import runner as runner_mod
from agents.runs.artifacts import (
    BLOCKLIST_FAILURE_REASON,
    blocklist_payload,
    write_summary_json,
    write_task_outputs,
)
from agents.runs.runner import TaskRunArtifacts, run_benchmark, run_single_task

# ---------- Pydantic 校验层 ------------------------------------------------------


def _write_minimal_config(path: Path, run_body: str) -> Path:
    """写一个最小可解析 YAML，把 run_body 插到 run: 下作为额外字段。"""
    path.write_text(f"agent:\n  model: m\n  api_base: http://x\n  api_key: k\nrun:\n{run_body}")
    return path


def test_write_summary_json_uses_utf8(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"

    write_summary_json(summary_path, {"failure_reason": "中文失败原因"})

    assert b"\xe4\xb8\xad\xe6\x96\x87" in summary_path.read_bytes()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {
        "failure_reason": "中文失败原因"
    }


@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param("  blocklist: 5\n", "blocklist must be a list", id="non-list"),
        pytest.param("  blocklist:\n    - 0\n", "task number must be >= 1", id="zero-or-negative"),
        pytest.param(
            "  blocklist:\n    - random_string\n",
            "must be an int or 'task_<n>'",
            id="unknown-string",
        ),
        # int 5 和 "task_5" 经规整后等价，应被识别为重复
        pytest.param(
            "  blocklist:\n    - 5\n    - task_5\n",
            "contains duplicate entries",
            id="duplicates-after-normalization",
        ),
        # bool 是 int 子类；显式 false → 0 会绕过 number 校验，所以单独排除
        pytest.param("  blocklist:\n    - true\n", "must be an int or string", id="bool"),
    ],
)
def test_blocklist_rejects_invalid_entries(tmp_path: Path, body: str, match: str) -> None:
    cfg_path = _write_minimal_config(tmp_path / "cfg.yaml", body)
    with pytest.raises(ValueError, match=match):
        load_app_config(cfg_path)


# ---------- runner 短路 ----------------------------------------------------------


def _make_config(*, tmp_path: Path, blocklist: tuple[str, ...] = ()) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "data"),
        agent=AgentConfig(model="m", api_base="http://x", api_key="primary-key"),
        run=RunConfig(
            output_dir=tmp_path / "runs",
            run_id=None,
            max_workers=2,
            task_timeout_seconds=60,
            blocklist=blocklist,
        ),
    )


def test_run_single_task_short_circuits_blocklisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """命中名单时不应触发子进程或同进程 core 路径。"""
    invoked = {"timeout": 0, "core": 0}

    def _fail_timeout(**_: Any) -> dict[str, Any]:
        invoked["timeout"] += 1
        raise AssertionError("timeout dispatcher must not be called for blocklisted task")

    def _fail_core(**_: Any) -> dict[str, Any]:
        invoked["core"] += 1
        raise AssertionError("core dispatcher must not be called for blocklisted task")

    monkeypatch.setattr(runner_mod, "run_single_task_with_timeout", _fail_timeout)
    monkeypatch.setattr(runner_mod, "run_single_task_core", _fail_core)

    cfg = _make_config(tmp_path=tmp_path, blocklist=("task_5",))
    run_output = tmp_path / "outdir"
    run_output.mkdir()

    artifact = run_single_task(task_id="task_5", config=cfg, run_output_dir=run_output)

    assert invoked == {"timeout": 0, "core": 0}
    assert artifact.succeeded is False
    assert artifact.failure_reason == BLOCKLIST_FAILURE_REASON
    assert artifact.prediction_csv_path is None


def test_run_single_task_runs_normally_when_not_blocklisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """blocklist 不命中时仍走原 timeout dispatcher——证明短路只在命中时触发。"""
    seen: list[str] = []

    def _ok_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
        del config
        seen.append(task_id)
        return {
            "task_id": task_id,
            "answer": None,
            "steps": [],
            "failure_reason": None,
            "succeeded": True,
        }

    monkeypatch.setattr(runner_mod, "run_single_task_with_timeout", _ok_timeout)

    cfg = _make_config(tmp_path=tmp_path, blocklist=("task_5",))
    run_output = tmp_path / "outdir"
    run_output.mkdir()

    artifact = run_single_task(task_id="task_7", config=cfg, run_output_dir=run_output)

    assert seen == ["task_7"]
    assert artifact.succeeded is True


# ---------- run_benchmark 整合 ---------------------------------------------------


class _CapturingRunSingleTask:
    """记录被实际调度的 task_id；返回成功占位 artifacts。"""

    def __init__(self) -> None:
        self.captured: list[str] = []

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
        del model, tools, answer_verifier
        # 模拟 run_single_task 内部的 blocklist 短路：保留行为以便 summary 统计正确
        if task_id in config.run.blocklist:
            return write_task_outputs(task_id, run_output_dir, blocklist_payload(task_id))
        self.captured.append(task_id)
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            succeeded=True,
            failure_reason=None,
        )


def test_run_benchmark_records_blocked_tasks_as_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset: Callable[[list[str]], None],
) -> None:
    """blocklist 内的任务必须出现在 summary 的 tasks 列表里且 succeeded=False。"""
    fake = _CapturingRunSingleTask()
    monkeypatch.setattr(runner_mod, "run_single_task", fake)
    patch_dataset(["task_1", "task_2", "task_3"])

    cfg = _make_config(tmp_path=tmp_path, blocklist=("task_2",))
    cfg = replace(cfg, run=replace(cfg.run, max_workers=1))

    run_output_dir, artifacts = run_benchmark(config=cfg)

    # 只有非 blocked 的任务真正调用过 dispatcher
    assert sorted(fake.captured) == ["task_1", "task_3"]
    # 但 summary 应包含全部 3 个任务
    assert len(artifacts) == 3
    by_id = {a.task_id: a for a in artifacts}
    assert by_id["task_2"].succeeded is False
    assert by_id["task_2"].failure_reason == BLOCKLIST_FAILURE_REASON
    assert by_id["task_1"].succeeded is True
    assert by_id["task_3"].succeeded is True

    # summary.json 同形：succeeded 计数排除 blocked
    summary_path = run_output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["task_count"] == 3
    assert summary["succeeded_task_count"] == 2
    blocked_entry = next(t for t in summary["tasks"] if t["task_id"] == "task_2")
    assert blocked_entry["failure_reason"] == BLOCKLIST_FAILURE_REASON
    assert blocked_entry["prediction_csv_path"] is None
