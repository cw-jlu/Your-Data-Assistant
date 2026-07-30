"""单任务运行的产物结构 + I/O 与 payload 帮手。

本模块只承担"形成磁盘产物"和"在状态字典间转换"两件事，不感知子进程编排。被
`runner.py` 与 `subprocess.py` 共同使用：

- `TaskRunArtifacts` / run-id 工厂 / `pin_key_on_config`：编排前需要的轻量帮手
- `timeout_payload` / `crash_payload`：失败兜底
- `write_task_outputs`：核心结果落盘成 prediction.csv

跨模块共享的符号统一不带前缀下划线（pyright strict 友好）；仅在本文件内部使用的
工具仍保留 `_xxx` 命名以表"模块私有"。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from agents.config import AppConfig, reject_placeholder_api_key


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    """单任务运行结束后，关于其产物的句柄集合。

    面向上层（CLI 展示、summary.json 序列化）：包含所有外部关心的字段，
    但不含 trace/answer 本体——那些已写进磁盘。
    """

    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None  # 未提交 answer 时为 None
    succeeded: bool  # Agent 是否正常调用 answer 工具
    failure_reason: str | None  # 失败时的可读原因

    def to_dict(self) -> dict[str, Any]:
        """序列化进 summary.json 的任务条目。

        Path → str：JSON 不原生支持 Path 类型。
        """
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path)
            if self.prediction_csv_path
            else None,
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }
        return payload


_RUN_ID_COUNTER_WIDTH = 3
_RUN_ID_COUNTER_MAX = 10**_RUN_ID_COUNTER_WIDTH - 1  # 999
_RUN_ID_ALLOC_RETRIES = 50


def _today_utc_prefix() -> str:
    """当前 UTC 日期的目录名前缀，如 `20260430-`。"""
    return datetime.now(UTC).strftime("%Y%m%d") + "-"


def create_run_id(output_root: Path) -> str:
    """生成 human-friendly run_id：`<UTC YYYYMMDD>-<NNN>`，例如 `20260430-001`。

    扫描 `output_root` 下今日已存在的子目录，取最大序号 +1；空目录或 output_root
    不存在时返回 `001`。仅做读侧扫描——不建目录；并发场景下两个进程可能同时
    算到同一序号，由 `create_run_output_dir` 的 `mkdir(exist_ok=False)` 仲裁，
    撞名时调用方应重试（已内建于 `create_run_output_dir`）。

    单日上限 999；超出抛 `RuntimeError`，提示用户显式指定 run_id 或更换 output_dir。
    日期分隔符 `-` 同时充当格式锚点：旧 22 字符微秒格式（无 `-`）天然不会与新格式
    冲突，老 run 目录原样保留。
    """
    prefix = _today_utc_prefix()
    max_existing = 0
    if output_root.is_dir():
        for entry in output_root.iterdir():
            name = entry.name
            if not entry.is_dir() or not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            # 严格匹配 NNN（与生成格式一致）：避免把误命名目录算进序号
            if len(suffix) == _RUN_ID_COUNTER_WIDTH and suffix.isdigit():
                value = int(suffix)
                if value > max_existing:
                    max_existing = value
    next_counter = max_existing + 1
    if next_counter > _RUN_ID_COUNTER_MAX:
        raise RuntimeError(
            f"daily run counter exceeded {_RUN_ID_COUNTER_MAX} on {prefix[:-1]}; "
            "set run.run_id explicitly or rotate output_dir."
        )
    return f"{prefix}{next_counter:0{_RUN_ID_COUNTER_WIDTH}d}"


def resolve_run_id(run_id: str | None, *, output_root: Path | None = None) -> str:
    """规整 run_id：None 时自动生成（需要 output_root 计数）；显式值需是合法单段目录名。

    拒绝包含 `/` `\\` 或等于 `.`/`..` 的输入，防止 Agent 误写到父目录或根目录。
    `output_root` 仅在 `run_id is None` 时使用：调用方若传了显式 id，扫描参数可省。
    """
    if run_id is None:
        if output_root is None:
            raise ValueError("output_root is required when run_id is auto-generated.")
        return create_run_id(output_root)

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    # 单级目录名约束：禁 `.`/`..`/含路径分隔符
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    """在 `output_root` 下创建 `<run_id>/` 子目录。

    显式 run_id：直接 mkdir(exist_ok=False)，撞名抛 FileExistsError——保持历史
    覆盖防护。

    自动 run_id：扫描 + mkdir + 撞名重试（最多 50 次）。并发 worker 可能同时算到
    同一序号，靠 mkdir 的原子失败仲裁，失败方重新扫描并取下一个序号。
    """
    if run_id is not None:
        effective_run_id = resolve_run_id(run_id)
        run_output_dir = output_root / effective_run_id
        run_output_dir.mkdir(parents=True, exist_ok=False)
        return effective_run_id, run_output_dir

    last_error: FileExistsError | None = None
    for _ in range(_RUN_ID_ALLOC_RETRIES):
        candidate = create_run_id(output_root)
        run_output_dir = output_root / candidate
        try:
            run_output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            last_error = exc
            continue
        return candidate, run_output_dir
    raise RuntimeError(
        f"failed to allocate a daily-counter run_id after {_RUN_ID_ALLOC_RETRIES} retries; "
        "filesystem may be saturated or counter exhausted."
    ) from last_error


def resolve_ratelimit_state_dir(config: AppConfig, effective_run_id: str) -> Path:
    """推导 `state_dir` 默认值：`<output_dir.parent>/ratelimit/<run_id>/`。

    默认 `output_dir = artifacts/runs`，所以默认状态目录是 `artifacts/ratelimit/<run_id>/`，
    与 `artifacts/runs/<run_id>/` 做兄弟目录——run 粒度隔离，避免跨 run 残留污染令牌桶。
    """
    return config.run.output_dir.parent / "ratelimit" / effective_run_id


def pin_key_on_config(config: AppConfig, *, task_index: int, effective_run_id: str) -> AppConfig:
    """返回 "单 Key 视图" 的子配置：按 task_index 轮询 api_keys 并写回 agent.api_key，
    同时把 rate_limit.state_dir 从 None 解成具体路径、把 run.run_id 从 None 解成真实值。

    用 `dataclasses.replace` 逐层生成新的 frozen dataclass；原 config 不变，可被多个
    子进程按不同 task_index 独立派生。

    若 `api_keys` 为空（未启用池），原样返回 config——对线上单 Key 路径无副作用。
    """
    if not config.agent.api_keys:
        return config
    key_index = task_index % len(config.agent.api_keys)
    picked = config.agent.api_keys[key_index]
    reject_placeholder_api_key(picked, field=f"agent.api_keys[{key_index}]")
    rl = config.agent.rate_limit
    if rl is not None and rl.state_dir is None:
        rl = replace(rl, state_dir=resolve_ratelimit_state_dir(config, effective_run_id))
    agent = replace(config.agent, api_key=picked, rate_limit=rl)
    # run_id 也落实：子进程 trace 里能看到真实 run_id，便于调试
    run = replace(config.run, run_id=effective_run_id)
    return replace(config, agent=agent, run=run)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON：UTF-8 + 2 缩进 + 行尾换行，保证 git diff 可读。"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    """写 prediction.csv（逐行 writer）。

    `newline=""` 按 csv 模块要求传，确保换行符跨平台一致（避免 Windows 写出 `\\r\\r\\n`）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    """构造 "任务彻底失败" 的占位结果，结构与 AgentRunResult.to_dict 同形。

    用于超时 / 子进程崩溃等情形，保证上层落盘/统计代码无分支。
    """
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


# 跳过名单 (run.blocklist) 命中时的 failure_reason；报表可按字面量过滤。
BLOCKLIST_FAILURE_REASON = "Task is in blocklist."


def blocklist_payload(task_id: str) -> dict[str, Any]:
    """构造 "任务被名单跳过" 的占位结果，供 runner 短路用。

    与超时/崩溃 payload 同形，只是 failure_reason 写死为 BLOCKLIST_FAILURE_REASON，
    让上层落盘 + summary 统计逻辑无须分支即可处理。
    """
    return _failure_run_result_payload(task_id, BLOCKLIST_FAILURE_REASON)


def _normalize_answer_payload(value: Any) -> dict[str, Any] | None:
    """把 trace 中的 answer-like 对象校验并规整成可写 CSV 的 payload。"""
    if not isinstance(value, dict):
        return None
    value_dict = cast(dict[str, Any], value)

    columns = value_dict.get("columns")
    rows = value_dict.get("rows")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(item, str) for item in cast(list[Any], columns))
    ):
        return None
    if not isinstance(rows, list):
        return None

    columns_list = cast(list[str], columns)
    rows_list = cast(list[Any], rows)
    normalized_rows: list[list[Any]] = []
    for row in rows_list:
        if not isinstance(row, list) or len(cast(list[Any], row)) != len(columns_list):
            return None
        normalized_rows.append(list(cast(list[Any], row)))
    return {"columns": list(columns_list), "rows": normalized_rows}


def timeout_payload(task_id: str, timeout_seconds: int) -> dict[str, Any]:
    """构造超时失败 payload。"""
    return _failure_run_result_payload(task_id, f"Task timed out after {timeout_seconds} seconds.")


def crash_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    """构造子进程崩溃失败 payload。"""
    return _failure_run_result_payload(task_id, failure_reason)


def write_task_outputs(
    task_id: str, run_output_dir: Path, run_result: dict[str, Any]
) -> TaskRunArtifacts:
    """把单任务结果落盘为 prediction.csv（仅当有 answer）。"""
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    prediction_csv_path: Path | None = None
    answer = _normalize_answer_payload(run_result.get("answer"))
    if answer is not None:
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            answer["columns"],
            answer["rows"],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def write_summary_json(summary_path: Path, payload: dict[str, Any]) -> None:
    """把整次 run 的汇总 dict 写成 summary.json，供 runner.run_benchmark 使用。"""
    _write_json(summary_path, payload)
