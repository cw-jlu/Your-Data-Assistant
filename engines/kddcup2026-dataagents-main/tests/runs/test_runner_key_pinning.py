"""runner 层的 Key 池轮询测试。

策略：
- 直接单测 `pin_key_on_config` / `build_model_adapter` 的 wrap 行为（纯函数，最干净）
- 用 monkeypatch 替换 `run_single_task` 为捕获 config 的 fake，跑 `run_benchmark`
  验证并行 / 串行 / CLI 单任务三条路径上 Key 确实被按 task_index 轮询固定
- 避免真的 fork 子进程或发 HTTP 请求——所有网络/进程侧效已被 fake 屏蔽
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agents.application import build_model_adapter
from agents.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    RateLimitConfig,
    RunConfig,
)
from agents.runs import runner as runner_mod
from agents.runs.artifacts import pin_key_on_config, resolve_ratelimit_state_dir
from agents.runs.runner import TaskRunArtifacts

# ------- 测试辅助：构造可用的 AppConfig ------------------------------------------


def _make_config(
    *,
    tmp_path: Path,
    api_keys: tuple[str, ...] = (),
    rate_limit: RateLimitConfig | None = None,
) -> AppConfig:
    """构造一个指向 tmp_path 的 AppConfig，用于 runner 测试。"""
    return AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "data"),
        agent=AgentConfig(
            model="m",
            api_base="http://x",
            api_key="primary-key",  # 单 Key 默认值
            api_keys=api_keys,
            rate_limit=rate_limit,
        ),
        run=RunConfig(
            output_dir=tmp_path / "runs",
            run_id=None,
            max_workers=4,
            task_timeout_seconds=60,
        ),
    )


# ------- pin_key_on_config ------------------------------------------------------


def test_pin_noop_when_api_keys_empty(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path=tmp_path)  # api_keys=()
    out = pin_key_on_config(cfg, task_index=0, effective_run_id="run-1")
    assert out is cfg  # 原样返回，不创建新对象


def test_pin_rotates_by_task_index(tmp_path: Path) -> None:
    rl = RateLimitConfig(state_dir=None)
    cfg = _make_config(tmp_path=tmp_path, api_keys=("key-A", "key-B"), rate_limit=rl)
    picked = [
        pin_key_on_config(cfg, task_index=i, effective_run_id="run-1").agent.api_key
        for i in range(4)
    ]
    assert picked == ["key-A", "key-B", "key-A", "key-B"]


def test_pin_rejects_placeholder_api_key(tmp_path: Path) -> None:
    rl = RateLimitConfig(state_dir=None)
    cfg = _make_config(tmp_path=tmp_path, api_keys=("YOUR_API_KEY_1",), rate_limit=rl)

    with pytest.raises(ValueError, match=r"agent\.api_keys\[0\].*placeholder"):
        pin_key_on_config(cfg, task_index=0, effective_run_id="run-1")


def test_pin_fills_state_dir_default(tmp_path: Path) -> None:
    rl = RateLimitConfig(state_dir=None)
    cfg = _make_config(tmp_path=tmp_path, api_keys=("key-A",), rate_limit=rl)
    out = pin_key_on_config(cfg, task_index=0, effective_run_id="run-42")
    assert out.agent.rate_limit is not None
    expected = resolve_ratelimit_state_dir(cfg, "run-42")
    assert out.agent.rate_limit.state_dir == expected


def test_pin_preserves_explicit_state_dir(tmp_path: Path) -> None:
    explicit = tmp_path / "my-ratelimit"
    rl = RateLimitConfig(state_dir=explicit)
    cfg = _make_config(tmp_path=tmp_path, api_keys=("key-A",), rate_limit=rl)
    out = pin_key_on_config(cfg, task_index=0, effective_run_id="run-42")
    assert out.agent.rate_limit is not None
    assert out.agent.rate_limit.state_dir == explicit  # 显式值不被覆盖


def test_pin_sets_run_id(tmp_path: Path) -> None:
    rl = RateLimitConfig(state_dir=None)
    cfg = _make_config(tmp_path=tmp_path, api_keys=("key-A",), rate_limit=rl)
    out = pin_key_on_config(cfg, task_index=0, effective_run_id="run-42")
    assert out.run.run_id == "run-42"


# ------- build_model_adapter wrap 行为 -------------------------------------------


def test_build_adapter_no_wrap_when_rate_limit_none(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path=tmp_path)  # rate_limit=None
    # NativeToolsOpenAIAdapter requires tools
    from agents.tools.registry import create_default_tool_registry

    tools = create_default_tool_registry()
    adapter = build_model_adapter(cfg, tools=tools)
    # 不应是 RateLimitedAdapter
    from agents.llm.rate_limit import RateLimitedAdapter

    assert not isinstance(adapter, RateLimitedAdapter)


def test_build_adapter_wraps_when_rate_limit_set(tmp_path: Path) -> None:
    rl = RateLimitConfig(state_dir=tmp_path / "rl")
    cfg = _make_config(tmp_path=tmp_path, api_keys=("k1",), rate_limit=rl)
    # pin 后 agent.api_key="k1"，rate_limit.state_dir 已被解析
    pinned = pin_key_on_config(cfg, task_index=0, effective_run_id="run-x")

    from agents.llm.rate_limit import RateLimitedAdapter
    from agents.tools.registry import create_default_tool_registry

    tools = create_default_tool_registry()
    adapter = build_model_adapter(pinned, tools=tools)
    assert isinstance(adapter, RateLimitedAdapter)
    assert adapter.api_key == "k1"


# ------- run_benchmark 层面的配置下推 --------------------------------------------


class _CapturingSingleTask:
    """替换 run_single_task 的 fake：只记录 (task_id, config.agent.api_key) 并返回占位 artifacts。"""

    def __init__(self) -> None:
        self.captured: list[tuple[str, str]] = []

    def __call__(
        self,
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        model: Any = None,
        tools: Any = None,
    ) -> TaskRunArtifacts:
        del model, tools
        self.captured.append((task_id, config.agent.api_key))
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            succeeded=True,
            failure_reason=None,
        )


def _patch_run_single_task(monkeypatch: pytest.MonkeyPatch) -> _CapturingSingleTask:
    fake = _CapturingSingleTask()
    monkeypatch.setattr(runner_mod, "run_single_task", fake)
    return fake


def test_run_benchmark_parallel_rotates_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset: Callable[[list[str]], None],
) -> None:
    fake = _patch_run_single_task(monkeypatch)
    patch_dataset(["task_0", "task_1", "task_2", "task_3"])

    rl = RateLimitConfig(state_dir=None)
    cfg = _make_config(tmp_path=tmp_path, api_keys=("key-A", "key-B"), rate_limit=rl)
    cfg = replace(cfg, run=replace(cfg.run, max_workers=2))

    runner_mod.run_benchmark(config=cfg)

    # 按 task_id 重排捕获（并行完成顺序不定）
    captured = dict(fake.captured)
    assert captured["task_0"] == "key-A"
    assert captured["task_1"] == "key-B"
    assert captured["task_2"] == "key-A"
    assert captured["task_3"] == "key-B"


def test_run_benchmark_serial_rotates_keys_when_pool_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset: Callable[[list[str]], None],
) -> None:
    fake = _patch_run_single_task(monkeypatch)
    patch_dataset(["task_0", "task_1", "task_2"])

    rl = RateLimitConfig(state_dir=None)
    cfg = _make_config(tmp_path=tmp_path, api_keys=("key-A", "key-B"), rate_limit=rl)
    cfg = replace(cfg, run=replace(cfg.run, max_workers=1))

    runner_mod.run_benchmark(config=cfg)

    # 串行按 task 顺序：索引 0→A, 1→B, 2→A
    keys_in_order = [key for _, key in fake.captured]
    assert keys_in_order == ["key-A", "key-B", "key-A"]


def test_run_benchmark_no_pool_passes_config_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset: Callable[[list[str]], None],
) -> None:
    fake = _patch_run_single_task(monkeypatch)
    patch_dataset(["task_0", "task_1"])

    cfg = _make_config(tmp_path=tmp_path)  # api_keys=()
    cfg = replace(cfg, run=replace(cfg.run, max_workers=2))

    runner_mod.run_benchmark(config=cfg)
    # 所有 task 都看到原 api_key（"primary-key"）
    keys = {key for _, key in fake.captured}
    assert keys == {"primary-key"}


def test_run_benchmark_state_dir_default_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_dataset: Callable[[list[str]], None],
) -> None:
    """未显式指定 state_dir 时，runner 应把它解析为 <output_dir.parent>/ratelimit/<run_id>/。"""

    captured_state_dirs: list[Path | None] = []

    def _capture_state_dir(
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        model: Any = None,
        tools: Any = None,
    ) -> TaskRunArtifacts:
        del model, tools
        rl = config.agent.rate_limit
        captured_state_dirs.append(rl.state_dir if rl else None)
        (run_output_dir / task_id).mkdir(parents=True, exist_ok=True)
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=run_output_dir / task_id,
            prediction_csv_path=None,
            succeeded=True,
            failure_reason=None,
        )

    monkeypatch.setattr(runner_mod, "run_single_task", _capture_state_dir)
    patch_dataset(["task_0"])

    rl = RateLimitConfig(state_dir=None)  # 让 runner 推导
    cfg = _make_config(tmp_path=tmp_path, api_keys=("key-A",), rate_limit=rl)
    cfg = replace(cfg, run=replace(cfg.run, max_workers=2, run_id="test-run"))

    runner_mod.run_benchmark(config=cfg)

    assert captured_state_dirs == [
        cfg.run.output_dir.parent / "ratelimit" / "test-run",
    ]


# ------- 单任务 CLI 路径 ---------------------------------------------------------


def test_run_single_task_cli_picks_first_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI `dabench run-task <id>` 的入口：有 api_keys 时必须选 index=0 那个 Key。"""
    captured_configs: list[AppConfig] = []

    def _fake_timeout_dispatch(*, task_id: str, config: AppConfig) -> dict[str, Any]:
        del task_id
        captured_configs.append(config)
        return {"task_id": "task_x", "answer": None, "steps": [], "succeeded": False}

    # patch 的是 runner.py `from agents.runs.subprocess import run_single_task_with_timeout` 拿到的
    # 本地绑定；为兼容旧测试保留的下划线别名 `_run_single_task_with_timeout` 不参与函数调用解析。
    monkeypatch.setattr(runner_mod, "run_single_task_with_timeout", _fake_timeout_dispatch)

    rl = RateLimitConfig(state_dir=None)
    cfg = _make_config(tmp_path=tmp_path, api_keys=("key-A", "key-B"), rate_limit=rl)
    run_output = tmp_path / "outdir"
    run_output.mkdir()
    runner_mod.run_single_task(task_id="task_x", config=cfg, run_output_dir=run_output)

    assert len(captured_configs) == 1
    assert captured_configs[0].agent.api_key == "key-A"
    # state_dir 也应被填充
    rl_out = captured_configs[0].agent.rate_limit
    assert rl_out is not None and rl_out.state_dir is not None


def test_run_single_task_cli_unchanged_without_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没配 api_keys 时，run_single_task 不碰 config（线上单 Key 路径）。"""
    captured: list[AppConfig] = []

    def _fake_timeout_dispatch(*, task_id: str, config: AppConfig) -> dict[str, Any]:
        del task_id
        captured.append(config)
        return {"task_id": "task_x", "answer": None, "steps": [], "succeeded": False}

    # patch 的是 runner.py `from agents.runs.subprocess import run_single_task_with_timeout` 拿到的
    # 本地绑定；为兼容旧测试保留的下划线别名 `_run_single_task_with_timeout` 不参与函数调用解析。
    monkeypatch.setattr(runner_mod, "run_single_task_with_timeout", _fake_timeout_dispatch)

    cfg = _make_config(tmp_path=tmp_path)  # 无 api_keys，无 rate_limit
    run_output = tmp_path / "outdir"
    run_output.mkdir()
    runner_mod.run_single_task(task_id="task_x", config=cfg, run_output_dir=run_output)

    assert len(captured) == 1
    assert captured[0].agent.api_key == "primary-key"
    assert captured[0].agent.rate_limit is None
