"""App configuration loading.

Precedence (highest wins):
  env var > YAML > dataclass default

Recognized env vars:
  MODEL_API_URL          → agent.api_base   (eval injects this)
  MODEL_API_KEY          → agent.api_key    (eval injects this)
  MODEL_NAME             → agent.model      (eval injects this)
  DABENCH_INPUT_DIR      → dataset.root_path
  DABENCH_OUTPUT_DIR     → run.output_dir
  DABENCH_RUN_ID         → run.run_id
  DABENCH_MAX_WORKERS    → run.max_workers
  DABENCH_TASK_TIMEOUT   → run.task_timeout_seconds
  DABENCH_FLAT_OUTPUT    → run.flat_output_dir   ("1"/"true"/"yes" → True)
  DABENCH_LOG_FILE       → run.log_file (where to mirror runtime logs)
  DABENCH_WALL_CLOCK_BUDGET → run.wall_clock_budget_seconds (int seconds; "0" disables)
  DABENCH_PYTHON_KERNEL_MODE → agent.python_kernel_mode ("persistent" | "ephemeral")
  DABENCH_SC_K              → agent.self_consistency_k
  DABENCH_SC_TEMP           → agent.self_consistency_temperature
  DABENCH_DISABLE_SELF_CONSISTENCY → forces k=1 when truthy (per-runtime opt-out)
  DABENCH_REPEAT_MAX        → run.repeat_max (max passes for cross-run voting)
  DABENCH_PASS_SAFETY_MARGIN → run.pass_safety_margin (next-pass time guard)

Env-var overlay matters for the eval container: judges inject
MODEL_API_URL/KEY/NAME at runtime and the YAML cannot have correct
placeholders ahead of time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


DEFAULT_MAX_STEPS_BY_DIFFICULTY: dict[str, int] = {
    # v5 S-5: easy bumped 8→12. task_11 / task_80 v4 forensic showed even
    # easy BIRD questions sometimes need fuzzy-match steps the 8-budget
    # didn't allow. Wall-clock impact is ~5% — easy tasks already finish
    # in well under their 300 s subprocess timeout.
    "easy": 12,
    "medium": 12,
    "hard": 24,
    "extreme": 32,
}


# Per-difficulty subprocess wall-clock budget. v4 holdout showed 4 / 50
# tasks hit the legacy 600 s timeout (extreme + large hard tier with
# >256 MB context) before producing an answer, costing those tasks 0
# points outright. Bumping hard / extreme buys enough headroom for
# heavier doc-grounded reasoning while leaving easy / medium below the
# previous default — the wall-clock governor still halves these on 8 h
# trigger so the 12 h envelope stays intact.
DEFAULT_TASK_TIMEOUT_BY_DIFFICULTY: dict[str, int] = {
    "easy": 300,
    "medium": 600,
    "hard": 900,
    "extreme": 1200,
}


# Single-model policy (team1438):
#   - Eval-time main solver is locked to qwen3.5-35b-a3b by the rules
#     (kddcup-rules-model). Auxiliary models are technically permitted
#     within the hardware budget, but our team policy is stricter:
#     **no auxiliary LLM, embedding, vision, or retrieval model is used
#     at any point.** Defaults below intentionally do NOT point at any
#     external provider — env / YAML must supply concrete endpoints.
@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0
    python_kernel_mode: str = "persistent"
    max_steps_by_difficulty: dict[str, int] | None = field(
        default_factory=lambda: dict(DEFAULT_MAX_STEPS_BY_DIFFICULTY),
    )
    # G-2 self-consistency: hard / extreme tier sample k times at higher
    # temperature and majority-vote on column_signature. k=1 collapses to
    # single ReActAgent execution. v5 S-1 makes each sample deterministic
    # via per-sample seed (sample_idx differentiated), so k samples now
    # explore k distinct reasoning paths reproducibly. Keep k=3 for budget
    # fit: with hard timeout 2400s and SC 0.85 budget, k=3 → ~680s/sample
    # which matches v4 hard-tier average duration.
    self_consistency_k: int = 3
    self_consistency_temperature: float = 0.5
    self_consistency_difficulties: tuple[str, ...] = ("hard", "extreme")


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600
    # Per-difficulty timeout override. When set, _run_single_task_with_timeout
    # picks `task_timeout_by_difficulty[difficulty]` and falls back to
    # `task_timeout_seconds` if the difficulty key is missing. Set to None
    # (or empty) to use the legacy single-value timeout for every task.
    task_timeout_by_difficulty: dict[str, int] | None = field(
        default_factory=lambda: dict(DEFAULT_TASK_TIMEOUT_BY_DIFFICULTY),
    )
    flat_output_dir: bool = False
    log_file: Path | None = None
    # Total wall-clock seconds for run_benchmark. Governor engages when the
    # remaining budget can no longer cover the remaining tasks at the
    # observed average per-task elapsed time. None disables the governor.
    wall_clock_budget_seconds: int | None = None
    # Multi-pass cross-run voting (H-1). repeat_max=1 collapses to the
    # legacy single-pass behaviour; >1 runs the benchmark up to that many
    # times into output_dir/_runs/run_<i>/ and votes the predictions into
    # output_dir/task_<id>/. The orchestrator stops early between passes
    # when (remaining_budget) < (last_pass_duration × pass_safety_margin),
    # so the first pass is always preserved as the fallback.
    repeat_max: int = 1
    pass_safety_margin: float = 1.3


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


_TRUE_LITERALS = frozenset({"1", "true", "yes", "on"})


def _path_value(raw_value: str | os.PathLike[str] | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _coerce_bool(raw: str | bool | None) -> bool:
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in _TRUE_LITERALS


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return value
    return None


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {}) or {}
    agent_payload = payload.get("agent", {}) or {}
    run_payload = payload.get("run", {}) or {}

    dataset_root_raw = _env_first("DABENCH_INPUT_DIR") or dataset_payload.get("root_path")
    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_root_raw, dataset_defaults.root_path),
    )

    model_raw = _env_first("MODEL_NAME") or agent_payload.get("model", agent_defaults.model)
    api_base_raw = _env_first("MODEL_API_URL") or agent_payload.get("api_base", agent_defaults.api_base)
    api_key_raw = _env_first("MODEL_API_KEY") or agent_payload.get("api_key", agent_defaults.api_key)
    kernel_mode_raw = _env_first("DABENCH_PYTHON_KERNEL_MODE") or agent_payload.get(
        "python_kernel_mode", agent_defaults.python_kernel_mode
    )
    mss_raw = agent_payload.get("max_steps_by_difficulty")
    if isinstance(mss_raw, dict):
        max_steps_by_difficulty: dict[str, int] | None = {
            str(k): int(v) for k, v in mss_raw.items()
        }
    elif mss_raw is None:
        # default: keep agent_defaults' factory-provided mapping
        max_steps_by_difficulty = agent_defaults.max_steps_by_difficulty
    else:
        # explicit non-dict (e.g. False / null in YAML) disables the override.
        max_steps_by_difficulty = None
    sc_k_raw = _env_first("DABENCH_SC_K") or agent_payload.get(
        "self_consistency_k", agent_defaults.self_consistency_k
    )
    sc_temp_raw = _env_first("DABENCH_SC_TEMP") or agent_payload.get(
        "self_consistency_temperature", agent_defaults.self_consistency_temperature
    )
    sc_disable_raw = os.environ.get("DABENCH_DISABLE_SELF_CONSISTENCY")
    sc_k_value = max(1, int(sc_k_raw))
    if _coerce_bool(sc_disable_raw):
        sc_k_value = 1

    sc_diff_raw = agent_payload.get("self_consistency_difficulties")
    if isinstance(sc_diff_raw, (list, tuple)):
        sc_difficulties = tuple(str(x) for x in sc_diff_raw)
    elif sc_diff_raw is None:
        sc_difficulties = agent_defaults.self_consistency_difficulties
    else:
        sc_difficulties = ()

    agent_config = AgentConfig(
        model=str(model_raw),
        api_base=str(api_base_raw),
        api_key=str(api_key_raw),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        python_kernel_mode=str(kernel_mode_raw),
        max_steps_by_difficulty=max_steps_by_difficulty,
        self_consistency_k=sc_k_value,
        self_consistency_temperature=float(sc_temp_raw),
        self_consistency_difficulties=sc_difficulties,
    )

    raw_run_id = _env_first("DABENCH_RUN_ID")
    if raw_run_id is None:
        raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    output_dir_raw = _env_first("DABENCH_OUTPUT_DIR") or run_payload.get("output_dir")
    max_workers_raw = _env_first("DABENCH_MAX_WORKERS")
    max_workers = int(max_workers_raw) if max_workers_raw else int(run_payload.get("max_workers", run_defaults.max_workers))
    timeout_raw = _env_first("DABENCH_TASK_TIMEOUT")
    task_timeout = int(timeout_raw) if timeout_raw else int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds))

    # Per-difficulty timeout override. Env DABENCH_TASK_TIMEOUT being set
    # signals an explicit single-value override → disable the per-difficulty
    # mapping so the env value takes full effect. Otherwise YAML-supplied
    # mapping (or the dataclass default) wins.
    if timeout_raw:
        task_timeout_by_difficulty: dict[str, int] | None = None
    else:
        ttbd_raw = run_payload.get("task_timeout_by_difficulty")
        if isinstance(ttbd_raw, dict):
            task_timeout_by_difficulty = {str(k): int(v) for k, v in ttbd_raw.items()}
        elif ttbd_raw is None:
            task_timeout_by_difficulty = run_defaults.task_timeout_by_difficulty
        else:
            task_timeout_by_difficulty = None

    flat_output_raw = os.environ.get("DABENCH_FLAT_OUTPUT")
    if flat_output_raw is None:
        flat_output = bool(run_payload.get("flat_output_dir", run_defaults.flat_output_dir))
    else:
        flat_output = _coerce_bool(flat_output_raw)

    log_file_raw = _env_first("DABENCH_LOG_FILE") or run_payload.get("log_file")
    log_file: Path | None = None
    if log_file_raw:
        log_file = Path(log_file_raw)
        if not log_file.is_absolute():
            log_file = (PROJECT_ROOT / log_file).resolve()

    budget_raw = _env_first("DABENCH_WALL_CLOCK_BUDGET")
    if budget_raw is None:
        budget_yaml = run_payload.get("wall_clock_budget_seconds", run_defaults.wall_clock_budget_seconds)
        wall_clock_budget: int | None = (
            int(budget_yaml) if budget_yaml not in (None, "") else None
        )
    else:
        budget_int = int(budget_raw)
        wall_clock_budget = budget_int if budget_int > 0 else None
    if wall_clock_budget is not None and wall_clock_budget <= 0:
        wall_clock_budget = None

    repeat_max_raw = _env_first("DABENCH_REPEAT_MAX") or run_payload.get(
        "repeat_max", run_defaults.repeat_max
    )
    pass_safety_raw = _env_first("DABENCH_PASS_SAFETY_MARGIN") or run_payload.get(
        "pass_safety_margin", run_defaults.pass_safety_margin
    )
    repeat_max_value = max(1, int(repeat_max_raw))
    pass_safety_value = float(pass_safety_raw)
    if pass_safety_value <= 1.0:
        # Margin must give us strictly more headroom than the next pass
        # itself; smaller values would risk overrunning the budget.
        pass_safety_value = run_defaults.pass_safety_margin

    run_config = RunConfig(
        output_dir=_path_value(output_dir_raw, run_defaults.output_dir),
        run_id=run_id,
        max_workers=max_workers,
        task_timeout_seconds=task_timeout,
        task_timeout_by_difficulty=task_timeout_by_difficulty,
        flat_output_dir=flat_output,
        log_file=log_file,
        wall_clock_budget_seconds=wall_clock_budget,
        repeat_max=repeat_max_value,
        pass_safety_margin=pass_safety_value,
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
