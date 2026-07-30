from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.paths import ENGINES_ROOT, RUNTIME_ROOT, uv_executable

TASK_ID_PATTERN = re.compile(r"^task_\d+$")


@dataclass(frozen=True, slots=True)
class EngineSpec:
    id: str
    name: str
    subtitle: str
    description: str
    strengths: tuple[str, ...]
    root_name: str
    accent: str
    supports_single: bool = True
    supports_batch: bool = True
    supports_experiment: bool = False

    @property
    def root(self) -> Path:
        return ENGINES_ROOT / self.root_name

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "subtitle": self.subtitle,
            "description": self.description,
            "strengths": list(self.strengths),
            "accent": self.accent,
            "supports_single": self.supports_single,
            "supports_batch": self.supports_batch,
            "supports_experiment": self.supports_experiment,
            "ready": self.root.is_dir() and uv_executable() is not None,
            "root": str(self.root),
        }


ENGINE_SPECS: tuple[EngineSpec, ...] = (
    EngineSpec(
        id="langgraph",
        name="LangGraph",
        subtitle="图工作流与多模态理解",
        description="显式状态图编排，带数据理解、歧义分析、过程校验和答案校验。",
        strengths=("LangGraph 状态机", "视频 ASR / 关键帧", "多级验证器"),
        root_name="kddcup-LangGraph",
        accent="#5eead4",
    ),
    EngineSpec(
        id="kobushi",
        name="Kobushi",
        subtitle="第四名决赛方案",
        description="PLAN → EXPLORE → ANSWER → VERIFY 分阶段 ReAct，可切换完整实验包。",
        strengths=("分阶段 ReAct", "投影与 SQL 防护", "在线音视频 ASR"),
        root_name="kddcup2026-data-agents-4th-place-solution-main",
        accent="#fb7185",
        supports_single=False,
        supports_experiment=True,
    ),
    EngineSpec(
        id="memory",
        name="Memory ReAct",
        subtitle="多轮投票与跨任务记忆",
        description="把自一致性、跨运行投票、错误模式和任务形状记忆组合到 ReAct 中。",
        strengths=("Self-consistency", "Cross-run vote", "错误模式记忆"),
        root_name="kddcup2026-data-agents-main",
        accent="#fbbf24",
    ),
    EngineSpec(
        id="mamba",
        name="Mamba Agent",
        subtitle="ETL 路由与原生工具调用",
        description="模块化 ETL、原生 tool calling、答案验证和 SQLite 全链路追踪。",
        strengths=("异构 ETL", "原生工具协议", "可观测追踪"),
        root_name="kddcup2026-dataagents-main",
        accent="#60a5fa",
    ),
)
ENGINE_BY_ID = {engine.id: engine for engine in ENGINE_SPECS}


@dataclass(slots=True)
class LaunchPlan:
    engine: EngineSpec
    command: list[str]
    cwd: Path
    env: dict[str, str]
    output_dir: Path
    config_path: Path | None = None


def _yaml_string(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _validated_request(payload: dict[str, Any], engine: EngineSpec) -> dict[str, Any]:
    dataset_root = Path(str(payload.get("dataset_root", ""))).expanduser().resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"数据集目录不存在：{dataset_root}")

    mode = str(payload.get("mode", "single"))
    if mode not in {"single", "batch"}:
        raise ValueError("mode 只能是 single 或 batch")
    if mode == "single" and not engine.supports_single:
        raise ValueError(f"{engine.name} 的提交入口只支持批量运行")

    task_id = str(payload.get("task_id", "")).strip()
    if mode == "single":
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError("单任务模式需要合法的 task_<数字> ID")
        if not (dataset_root / task_id).is_dir():
            raise ValueError(f"数据集中找不到 {task_id}")

    model = str(payload.get("model", "")).strip()
    api_base = str(payload.get("api_base", "")).strip()
    api_key = str(payload.get("api_key", "")).strip()
    if not model or not api_base:
        raise ValueError("模型名称和 API 地址不能为空")

    def bounded_int(name: str, default: int, lower: int, upper: int) -> int:
        try:
            value = int(payload.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须是整数") from exc
        if value < lower or value > upper:
            raise ValueError(f"{name} 必须在 {lower}–{upper} 之间")
        return value

    return {
        "dataset_root": dataset_root,
        "mode": mode,
        "task_id": task_id,
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "max_workers": bounded_int("max_workers", 2, 1, 32),
        "max_steps": bounded_int("max_steps", 16, 1, 128),
        "timeout": bounded_int("timeout", 900, 30, 21600),
        "experiment": str(payload.get("experiment", "exp_154_v1_audio_asr")).strip()
        or "exp_154_v1_audio_asr",
    }


def _common_env(values: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "MODEL_NAME": values["model"],
            "MODEL_API_URL": values["api_base"],
            "MODEL_API_KEY": values["api_key"],
            "NO_COLOR": "1",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _cli_command(values: dict[str, Any], config_path: Path) -> list[str]:
    command = [uv_executable() or "uv", "run", "dabench"]
    if values["mode"] == "single":
        command.extend(["run-task", values["task_id"]])
    else:
        command.append("run-benchmark")
    command.extend(["--config", str(config_path)])
    return command


def _write_config(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def build_launch_plan(engine_id: str, payload: dict[str, Any], run_id: str) -> LaunchPlan:
    engine = ENGINE_BY_ID.get(engine_id)
    if engine is None:
        raise ValueError(f"未知引擎：{engine_id}")
    if not engine.root.is_dir():
        raise ValueError(f"引擎源码不存在：{engine.root}")
    if uv_executable() is None:
        raise ValueError("未找到 uv，请先安装 uv")

    values = _validated_request(payload, engine)
    config_path = RUNTIME_ROOT / "configs" / f"{run_id}.yaml"
    output_root = RUNTIME_ROOT / "outputs"
    output_dir = output_root / run_id
    env = _common_env(values)

    if engine.id == "langgraph":
        env.update(
            {
                "UNIFIED_MODEL": values["model"],
                "UNIFIED_API_BASE": values["api_base"],
                "UNIFIED_API_KEY": values["api_key"],
            }
        )
        config = f"""
dataset:
  root_path: {_yaml_string(values["dataset_root"])}
agent:
  model: ""
  model_env: UNIFIED_MODEL
  api_base: ""
  api_base_env: UNIFIED_API_BASE
  api_key: ""
  api_key_env: UNIFIED_API_KEY
  max_steps: {values["max_steps"]}
  temperature: 0.0
run:
  output_dir: {_yaml_string(output_root)}
  run_id: {_yaml_string(run_id)}
  max_workers: {values["max_workers"]}
  task_timeout_seconds: {values["timeout"]}
"""
        _write_config(config_path, config)
        command = _cli_command(values, config_path)
    elif engine.id == "memory":
        config = f"""
dataset:
  root_path: {_yaml_string(values["dataset_root"])}
agent:
  model: ""
  api_base: ""
  api_key: ""
  max_steps: {values["max_steps"]}
  temperature: 0.0
run:
  output_dir: {_yaml_string(output_root)}
  run_id: {_yaml_string(run_id)}
  max_workers: {values["max_workers"]}
  task_timeout_seconds: {values["timeout"]}
"""
        _write_config(config_path, config)
        command = _cli_command(values, config_path)
    elif engine.id == "mamba":
        # This upstream loader has no environment overlay. The temporary file is
        # removed by RunManager as soon as the subprocess exits.
        config = f"""
dataset:
  root_path: {_yaml_string(values["dataset_root"])}
agent:
  model: {_yaml_string(values["model"])}
  api_base: {_yaml_string(values["api_base"])}
  api_key: {_yaml_string(values["api_key"])}
  max_steps: {values["max_steps"]}
run:
  output_dir: {_yaml_string(output_root)}
  run_id: {_yaml_string(run_id)}
  max_workers: {values["max_workers"]}
  task_timeout_seconds: {values["timeout"]}
tracing:
  enabled: true
  db_path: {_yaml_string(RUNTIME_ROOT / "traces" / f"{run_id}.db")}
"""
        _write_config(config_path, config)
        command = _cli_command(values, config_path)
    else:
        config_path = None
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = RUNTIME_ROOT / "logs" / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        env.update(
            {
                "KDD_INPUT": str(values["dataset_root"]),
                "KDD_OUTPUT": str(output_dir),
                "KDD_LOGS": str(log_dir),
                "EXPERIMENT_NAME": values["experiment"],
                "MAX_WORKERS": str(values["max_workers"]),
                "TASK_TIMEOUT_SECONDS": str(values["timeout"]),
            }
        )
        command = [uv_executable() or "uv", "run", "python", "submission/main.py"]

    return LaunchPlan(
        engine=engine,
        command=command,
        cwd=engine.root,
        env=env,
        output_dir=output_dir,
        config_path=config_path,
    )


def list_tasks(dataset_root: str) -> list[dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"数据集目录不存在：{root}")

    tasks: list[dict[str, Any]] = []
    for task_dir in sorted(
        (path for path in root.iterdir() if path.is_dir() and TASK_ID_PATTERN.fullmatch(path.name)),
        key=lambda path: int(path.name.split("_", 1)[1]),
    ):
        task_json = task_dir / "task.json"
        metadata: dict[str, Any] = {}
        if task_json.is_file():
            try:
                metadata = json.loads(task_json.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                metadata = {}
        context = task_dir / "context"
        file_count = sum(1 for path in context.rglob("*") if path.is_file()) if context.is_dir() else 0
        tasks.append(
            {
                "id": task_dir.name,
                "difficulty": str(metadata.get("difficulty", "unknown")),
                "question": str(metadata.get("question", "")),
                "file_count": file_count,
            }
        )
    return tasks
