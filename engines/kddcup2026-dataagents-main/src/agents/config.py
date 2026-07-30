"""应用配置模块。

负责把 YAML 配置文件读成 `AppConfig`，并把可选字段退化为合理的默认值，
使得上层（CLI、runner、ReAct agent）不需要关心缺省处理。

配置类本身是冻结 Pydantic dataclass：
- 运行时仍然拿到 `AppConfig` / `AgentConfig` 等 dataclass，可继续用
  `dataclasses.replace` 派生不可变配置。
- YAML 校验直接走同一套类型，避免运行时 dataclass 与独立 Pydantic model
  双重定义。
- `bool` / `int` 字段使用 `StrictBool` / `StrictInt`，
  关闭 YAML 字符串到 bool/int 的隐式转换（详见 design.md D6 与 CHANGELOG）。
- 校验失败时取首条错误并剥去 `"Value error, "` 前缀，再用 `ValueError` 抛出，
  保持调用方看到的是普通配置错误。
"""

from __future__ import annotations

import os
import warnings
from dataclasses import field
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import (
    ConfigDict,
    StrictBool,
    StrictInt,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.dataclasses import dataclass

# 项目根目录：由本文件所在位置向上两级得到
# 注：src/agents/config.py → parents[2] = 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ETL_SCRATCH_ROOT = Path(os.environ.get("DABENCH_SCRATCH_ROOT", "/tmp/dabench"))


def _default_dataset_root() -> Path:
    """数据集根目录的默认值：`<project>/data/public/input`。"""
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    """运行产物的默认落盘根目录：`<project>/artifacts/runs`。"""
    return PROJECT_ROOT / "artifacts" / "runs"


# 后端类型字面量：决定思考模式参数在 extra_body 里的位置
# - "vllm":      enable_thinking 嵌套到 chat_template_kwargs（vLLM `--reasoning-parser qwen3` 要求）
# - "dashscope": enable_thinking 走顶层（DashScope OpenAI 兼容模式约定）
# - "deepseek":  thinking 对象走 extra_body 顶层（DeepSeek V4 API 协议）
# None = 由 adapter 按 api_base 主机名自动推断（含 aliyuncs.com/dashscope → dashscope，
#        api.deepseek.com → deepseek）。
BackendKindLiteral = Literal["vllm", "dashscope", "deepseek"]

_PYDANTIC_DATACLASS_CONFIG = ConfigDict(extra="forbid")

# RateLimitConfig.reserve_output_tokens 校验器借助 context 拿到父级 max_tokens，
# 保持 "reserve > 0" 与 "reserve <= max_tokens" 在同一个校验器里相邻执行。
_CONTEXT_KEY_MAX_TOKENS = "_max_tokens"


def is_placeholder_api_key(api_key: str) -> bool:
    """Return whether *api_key* is one of the checked-in example placeholders."""
    normalized = api_key.strip().upper()
    return normalized == "YOUR_API_KEY" or normalized.startswith("YOUR_API_KEY_")


def reject_placeholder_api_key(api_key: str, *, field: str) -> None:
    """Fail before a placeholder key can be pinned or passed to an adapter."""
    if is_placeholder_api_key(api_key):
        raise ValueError(
            f"{field} contains placeholder {api_key!r}; replace it with a real API key "
            "before running, or remove agent.api_keys/rate_limit to use agent.api_key."
        )


def _resolve_anchor_path(raw: Any, default: Path) -> Path:
    """模仿旧 `_path_value`：falsy → default；绝对 → 原样；相对 → PROJECT_ROOT 解析。"""
    if not raw:
        return default
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _resolve_state_dir(raw: Any) -> Path | None:
    """`rate_limit.state_dir` 专用：空/None → None；不调 `.resolve()` 与旧实现一致。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    candidate = Path(text)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


@dataclass(frozen=True, slots=True, config=_PYDANTIC_DATACLASS_CONFIG)
class DatasetConfig:
    """数据集相关配置。

    仅包含输入数据的根路径，其下应按 `task_<id>/` 组织每个任务。
    """

    # 公开数据集 input 目录；YAML 里缺省时回落到 PROJECT_ROOT/data/public/input
    root_path: Path = field(default_factory=_default_dataset_root)

    @field_validator("root_path", mode="before")
    @classmethod
    def _resolve_root_path(cls, v: Any) -> Path:
        return _resolve_anchor_path(v, _default_dataset_root())


@dataclass(frozen=True, slots=True, config=_PYDANTIC_DATACLASS_CONFIG)
class RateLimitConfig:
    """本地调限速 LLM 时的速率限制参数（RPM + TPM 双桶）。

    只在 `AgentConfig.rate_limit` 被显式配置时启用；线上 vLLM 路径请留空以保持 PR-1 行为。

    - rpm_per_key / tpm_per_key: 单 Key 的名义上限（与上游账号配额一致）
    - safety_factor: 实际容量 = cap * safety_factor，留出突发/时钟抖动的 buffer（0 < f <= 1）
    - reserve_output_tokens: 调用前按 "estimate_prompt + reserve_output" 预扣 TPM；
      实际 usage 回来后按差值补扣或退款，因此这里给的是输出上限的近似值（与 max_tokens 对齐）
    - max_retries: 429/5xx 重试次数上限（不含首次），>=0
    - backoff_initial_seconds / backoff_max_seconds: 指数退避起步和封顶；若响应带 Retry-After 则优先用 header
    - state_dir: 状态/锁文件根目录；None 时由 runner 自动推导为 `<artifacts_root>/ratelimit/<run_id>/`
    """

    rpm_per_key: StrictInt = 600
    tpm_per_key: StrictInt = 1_000_000
    safety_factor: float = 0.85
    reserve_output_tokens: StrictInt = 16384
    max_retries: StrictInt = 6
    backoff_initial_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    state_dir: Path | None = None

    @field_validator("rpm_per_key")
    @classmethod
    def _check_rpm(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("agent.rate_limit.rpm_per_key must be positive.")
        return v

    @field_validator("tpm_per_key")
    @classmethod
    def _check_tpm(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("agent.rate_limit.tpm_per_key must be positive.")
        return v

    @field_validator("safety_factor")
    @classmethod
    def _check_safety_factor(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError("agent.rate_limit.safety_factor must be in the interval (0, 1].")
        return v

    @field_validator("reserve_output_tokens")
    @classmethod
    def _check_reserve(cls, v: int, info: ValidationInfo) -> int:
        if v <= 0:
            raise ValueError("agent.rate_limit.reserve_output_tokens must be positive.")
        ctx: dict[str, Any] = info.context or {}
        max_tokens = ctx.get(_CONTEXT_KEY_MAX_TOKENS, 0)
        if isinstance(max_tokens, int) and max_tokens > 0 and v > max_tokens:
            raise ValueError(
                "agent.rate_limit.reserve_output_tokens must not exceed agent.max_tokens."
            )
        return v

    @field_validator("max_retries")
    @classmethod
    def _check_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("agent.rate_limit.max_retries must be >= 0.")
        return v

    @field_validator("backoff_initial_seconds")
    @classmethod
    def _check_backoff_initial(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("agent.rate_limit.backoff_initial_seconds must be positive.")
        return v

    @field_validator("state_dir", mode="before")
    @classmethod
    def _resolve_state_dir_field(cls, v: Any) -> Path | None:
        return _resolve_state_dir(v)

    @model_validator(mode="after")
    def _check_backoff_max(self) -> RateLimitConfig:
        if self.backoff_max_seconds < self.backoff_initial_seconds:
            raise ValueError(
                "agent.rate_limit.backoff_max_seconds must be >= backoff_initial_seconds."
            )
        return self


@dataclass(frozen=True, slots=True, config=_PYDANTIC_DATACLASS_CONFIG)
class AgentConfig:
    """ReAct Agent 及底层模型调用的配置。

    api_key 作为**明文**从 YAML 读取；生产部署应通过环境变量或 secret store 注入,
    而不是写进版本化的配置文件。

    max_tokens 用于 chat.completions 请求的同名字段（总输出上限，含 reasoning tokens）：
    - 0:   不发送该字段，沿用服务端默认（线上 vLLM 一般不限）
    - >0:  显式下发。本地调用百炼 Qwen3.5 限 TPM 的场景下建议填，
           预扣 token 误差会小很多；同时兜住单次 runaway 生成

    max_empty_tool_call_retries 是 native 协议下的空 tool_calls 独立重试预算：
    空 tool_calls 会写纠偏 observation，但不消耗业务 max_steps；达到该上限后失败，
    避免模型 reasoning-only 无限循环。

    enable_thinking 控制思考模式（Qwen3 特有；经 OpenAI SDK `extra_body` 透传）：
    - None:  不发送该字段，沿用服务端默认（vLLM `--reasoning-parser qwen3` 为 on）
    - True:  显式开；与线上行为对齐
    - False: 显式关；本地省 token 时可选，但线上不可用→慎用

    backend_kind 决定思考模式参数在 extra_body 里的放置位置（各后端协议不一致）：
    - None (默认):  由 adapter 按 api_base 主机名自动推断
                    （含 aliyuncs.com/dashscope → dashscope，api.deepseek.com → deepseek，
                    其余 → vllm）
    - "vllm":       嵌套到 chat_template_kwargs.enable_thinking
                    （vLLM `--reasoning-parser qwen3` 要求；顶层会被静默忽略）
    - "dashscope":  顶层 enable_thinking（DashScope OpenAI 兼容模式约定）
    - "deepseek":   顶层 thinking 对象（DeepSeek V4 API 协议）
    误判时显式声明即可覆盖。

    thinking_mode / reasoning_effort 是 DeepSeek V4 专用的思考模式控制：
    - 与 enable_thinking（Qwen3 专用）互斥；两者同时非 None 会 load 时报错
    - thinking_mode="enabled"/"disabled" → extra_body.thinking.type
    - reasoning_effort="high"/"max" → extra_body.thinking.reasoning_effort
    - DeepSeek V4 thinking 默认 enabled，默认 effort 为 high
    - thinking enabled 时 temperature/top_p 被服务端忽略（可设但无效）

    采样字段（temperature / top_p / top_k / min_p / presence_penalty / seed）保存
    "用户原始意图"：None = 未显式设置，由 `application.py:_resolve_sampling_defaults` 按
    `enable_thinking` 选 Qwen3 官方推荐组合：
    - thinking=True:  temperature=0.6 / top_p=0.95 / top_k=20 / min_p=0 / presence_penalty=0
    - thinking=False: temperature=0.7 / top_p=0.80 / top_k=20 / min_p=0 / presence_penalty=1.5
    - thinking=None:  全部不下发（与 enable_thinking=None 的语义对偶）
    用户显式设置某字段时，该字段无论 mode 都原样下发，不会被默认值覆盖。
    Qwen3 官方在 thinking 模式下显式反推荐 greedy decoding（temperature=0.0）；
    用户硬写 0.0 仍允许，但加载时会触发 UserWarning。

    api_keys / rate_limit 是 PR-2 新增的 "本地 Key 池 + 限速" 入口：
    - 两者都为空/None（默认）: 走单 Key 直连路径，与 PR-1 行为一致
    - 两者均配置:             runner 按 task_index 轮询 api_keys，并用 RateLimitedAdapter 包壳
    - 只有其一:               load 时报错，避免 "只池不限" / "只限无池" 的误配
    """

    model: str = "gpt-4.1-mini"  # OpenAI 兼容 chat.completions 的模型标识符
    api_base: str = "https://api.openai.com/v1"  # 兼容 OpenAI 协议的推理服务根地址
    api_key: str = ""  # 认证 Key；空串会在实际调用时抛 RuntimeError
    max_steps: StrictInt = 16  # 单任务 ReAct 循环上限，防止无限思考
    # 采样字段：None = 未显式设置 → 由 _resolve_sampling_defaults 按 mode 兜底
    temperature: float | None = None
    top_p: float | None = None
    top_k: StrictInt | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    seed: StrictInt | None = None
    # 原生协议下，是否允许模型一轮返回多个 tool_calls；个别 OpenAI-兼容后端不支持该字段，可关闭
    allow_parallel_tool_calls: StrictBool = True
    max_tokens: StrictInt = 16384  # 单次输出上限；0=不下发（用服务端默认）
    enable_thinking: StrictBool | None = None  # None=不下发；True/False=走 extra_body 透传
    # DeepSeek V4 thinking 参数（与 Qwen3 enable_thinking 互斥使用）：
    # thinking_mode 控制 DeepSeek V4 的推理模式：
    # - None:       不下发 thinking 对象（沿用服务端默认，V4 默认 enabled）
    # - "enabled":  显式开启 thinking
    # - "disabled": 显式关闭 thinking
    thinking_mode: Literal["enabled", "disabled"] | None = None
    # reasoning_effort 控制 thinking 的深度（仅 thinking_mode="enabled" 时有意义）：
    # - None:   不下发（沿用服务端默认 high）
    # - "high": 默认推理深度
    # - "max":  最大推理深度
    reasoning_effort: Literal["high", "max"] | None = None
    # 后端类型（决定思考模式参数的放置位置）；None=按 api_base 推断
    backend_kind: BackendKindLiteral | None = None
    # PR-2：本地 Key 池；runner 按 task_index 轮询并下推单 Key 视图给子进程。空元组=禁用
    api_keys: tuple[str, ...] = ()
    # PR-2：限速配置；None=不启用限速器（线上 vLLM 路径默认如此）
    rate_limit: RateLimitConfig | None = None
    max_empty_tool_call_retries: StrictInt = 8
    enable_preact: StrictBool = False
    enable_answer_verifier: StrictBool = True
    # Video sub-agent report tool_choice mode:
    # - "auto" (default): works with thinking enabled everywhere; DashScope compatible-mode
    #   rejects named tool_choice in thinking mode (400).
    # - "forced": opt-in optimization — request-level named-function tool_choice pins `report`
    #   in one turn; only for deployments verified via scripts/smoke_video_tool_choice.py.
    video_tool_choice: Literal["forced", "auto"] = "auto"
    # Video sub-agent can need a larger reasoning+tool-call budget than the main ReAct loop.
    # None = inherit max_tokens; 0 = do not send max_tokens for video requests.
    video_max_tokens: StrictInt | None = None
    # Video sub-agent thinking mode override.
    # None = inherit enable_thinking from the main agent; True/False = explicit override.
    video_enable_thinking: StrictBool | None = None

    @field_validator("model", "api_base", "api_key", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        # 旧实现 `str(...)`：保留 `123 → "123"` 与 `null → "None"`，避免对历史 YAML 静默破坏。
        return str(v)

    @field_validator("max_tokens", "video_max_tokens")
    @classmethod
    def _check_max_tokens(cls, v: int | None, info: ValidationInfo) -> int | None:
        if v is None:
            return v
        if v < 0:
            raise ValueError(f"agent.{info.field_name} must be >= 0.")
        return v

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (0.0 <= v <= 2.0):
            raise ValueError("agent.temperature must be in the interval [0.0, 2.0].")
        return v

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (0.0 < v <= 1.0):
            raise ValueError("agent.top_p must be in the interval (0.0, 1.0].")
        return v

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if v < 1:
            raise ValueError("agent.top_k must be >= 1.")
        return v

    @field_validator("min_p")
    @classmethod
    def _check_min_p(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (0.0 <= v <= 1.0):
            raise ValueError("agent.min_p must be in the interval [0.0, 1.0].")
        return v

    @field_validator("presence_penalty")
    @classmethod
    def _check_presence_penalty(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not (-2.0 <= v <= 2.0):
            raise ValueError("agent.presence_penalty must be in the interval [-2.0, 2.0].")
        return v

    @field_validator("enable_thinking", mode="before")
    @classmethod
    def _check_enable_thinking(cls, v: Any) -> Any:
        if v is None or isinstance(v, bool):
            return v
        raise ValueError(f"agent.enable_thinking must be a bool or null; got {type(v).__name__}.")

    @field_validator("backend_kind", mode="before")
    @classmethod
    def _normalize_backend_kind(cls, v: Any) -> Any:
        if v is None:
            return v
        normalized = str(v).strip().lower()
        if normalized not in {"vllm", "dashscope", "deepseek"}:
            raise ValueError(
                f"agent.backend_kind must be 'vllm', 'dashscope', 'deepseek', or null; "
                f"got {normalized!r}."
            )
        return normalized

    @field_validator("api_keys", mode="before")
    @classmethod
    def _check_api_keys(cls, v: Any) -> tuple[str, ...]:
        if v is None:
            return ()
        if not isinstance(v, (list, tuple)):
            raise ValueError("agent.api_keys must be a list of strings.")
        items = cast(list[Any] | tuple[Any, ...], v)
        parsed: list[str] = []
        for idx, item in enumerate(items):
            if not isinstance(item, str):
                raise ValueError(
                    f"agent.api_keys[{idx}] must be a string; got {type(item).__name__}."
                )
            stripped = item.strip()
            if not stripped:
                raise ValueError(f"agent.api_keys[{idx}] must be a non-empty string.")
            parsed.append(stripped)
        if len(parsed) != len(set(parsed)):
            raise ValueError("agent.api_keys contains duplicate entries.")
        return tuple(parsed)

    @field_validator("rate_limit", mode="before")
    @classmethod
    def _validate_rate_limit(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None:
            return None
        max_tokens_value = info.data.get("max_tokens", 16384)
        if isinstance(v, RateLimitConfig):
            if (
                isinstance(max_tokens_value, int)
                and max_tokens_value > 0
                and v.reserve_output_tokens > max_tokens_value
            ):
                raise ValueError(
                    "agent.rate_limit.reserve_output_tokens must not exceed agent.max_tokens."
                )
            return v
        if not isinstance(v, dict):
            raise ValueError("agent.rate_limit must be a mapping.")
        return TypeAdapter(RateLimitConfig).validate_python(
            v, context={_CONTEXT_KEY_MAX_TOKENS: max_tokens_value}
        )

    @field_validator("max_empty_tool_call_retries")
    @classmethod
    def _check_max_empty(cls, v: int) -> int:
        if v < 0:
            raise ValueError("agent.max_empty_tool_call_retries must be >= 0.")
        return v

    @model_validator(mode="after")
    def _warn_thinking_with_greedy(self) -> AgentConfig:
        if self.enable_thinking is True and self.temperature == 0.0:
            warnings.warn(
                "agent.temperature=0.0 with enable_thinking=true uses greedy decoding, "
                "which the Qwen3 model card explicitly discourages "
                "(performance degradation and endless repetitions). "
                "Consider temperature=0.6 (Qwen3 thinking-mode recommendation) "
                "or unset temperature to use the mode-aware default.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _check_thinking_mutual_exclusion(self) -> AgentConfig:
        if self.enable_thinking is not None and self.thinking_mode is not None:
            raise ValueError(
                "agent.enable_thinking (Qwen3) and agent.thinking_mode (DeepSeek) "
                "are mutually exclusive; set one and leave the other null."
            )
        if self.reasoning_effort is not None and self.thinking_mode != "enabled":
            raise ValueError("agent.reasoning_effort requires agent.thinking_mode='enabled'.")
        return self


@dataclass(frozen=True, slots=True, config=_PYDANTIC_DATACLASS_CONFIG)
class RunConfig:
    """批量运行（run-benchmark / run-task）的调度与产物落盘配置。"""

    output_dir: Path = field(default_factory=_default_run_output_dir)  # 运行产物落盘根目录
    run_id: str | None = None  # 显式指定的 run 子目录名；None 时按 UTC 时间戳生成
    max_workers: StrictInt = 4  # 并行线程数（ThreadPoolExecutor 大小）
    task_timeout_seconds: StrictInt = 600  # 单任务硬超时，超时会 kill 子进程
    # 跳过名单：列出的 task_<n> 不会被实际执行，runner 直接产出失败 artifact（succeeded=False）。
    # 用途：临时屏蔽已知会卡住超时桶或正在调试的任务，避免污染整次 run 的 wall-clock 与限速预算。
    blocklist: tuple[str, ...] = ()

    @field_validator("output_dir", mode="before")
    @classmethod
    def _resolve_output_dir(cls, v: Any) -> Path:
        return _resolve_anchor_path(v, _default_run_output_dir())

    @field_validator("run_id", mode="before")
    @classmethod
    def _normalize_run_id(cls, v: Any) -> str | None:
        if v is None:
            return None
        normalized = str(v).strip()
        return normalized or None

    @field_validator("blocklist", mode="before")
    @classmethod
    def _check_blocklist(cls, v: Any) -> tuple[str, ...]:
        # 与 cli.py `_coerce_task_number` 对齐：接受 int / "5" / "task_5"，全部归一化到 "task_<n>"。
        if v is None:
            return ()
        if not isinstance(v, (list, tuple)):
            raise ValueError("run.blocklist must be a list of task ids.")
        items = cast(list[Any] | tuple[Any, ...], v)
        normalized: list[str] = []
        for idx, item in enumerate(items):
            if isinstance(item, bool) or not isinstance(item, (int, str)):
                raise ValueError(
                    f"run.blocklist[{idx}] must be an int or string; got {type(item).__name__}."
                )
            if isinstance(item, int):
                number = item
            else:
                cleaned = item.strip()
                if not cleaned:
                    raise ValueError(f"run.blocklist[{idx}] must be a non-empty string.")
                token = cleaned.removeprefix("task_") if cleaned.startswith("task_") else cleaned
                try:
                    number = int(token)
                except ValueError as exc:
                    raise ValueError(
                        f"run.blocklist[{idx}] must be an int or 'task_<n>'; got {item!r}."
                    ) from exc
            if number < 1:
                raise ValueError(f"run.blocklist[{idx}] task number must be >= 1; got {number}.")
            normalized.append(f"task_{number}")
        if len(normalized) != len(set(normalized)):
            raise ValueError("run.blocklist contains duplicate entries.")
        return tuple(normalized)


def _default_traces_db() -> Path:
    return PROJECT_ROOT / "artifacts" / "traces.db"


@dataclass(frozen=True, slots=True, config=_PYDANTIC_DATACLASS_CONFIG)
class TracingConfig:
    """Structured tracing configuration (opt-in)."""

    enabled: StrictBool = False
    db_path: Path = field(default_factory=_default_traces_db)

    @field_validator("db_path", mode="before")
    @classmethod
    def _resolve_db_path(cls, v: Any) -> Path:
        return _resolve_anchor_path(v, _default_traces_db())


@dataclass(frozen=True, slots=True, config=_PYDANTIC_DATACLASS_CONFIG)
class AppConfig:
    """顶层聚合配置，`load_app_config` 返回的完整对象。"""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)
    tracing: TracingConfig | None = None


_APP_CONFIG_ADAPTER = TypeAdapter(AppConfig)


def _check_loaded_config(config: AppConfig) -> AppConfig:
    """Apply YAML-level invariants that direct runtime constructors historically allowed."""
    has_keys = bool(config.agent.api_keys)
    has_rate_limit = config.agent.rate_limit is not None
    if has_keys != has_rate_limit:
        raise ValueError(
            "agent.api_keys and agent.rate_limit must be configured together; "
            "leave both unset for the single-key / no-limit default path."
        )
    return config


def load_app_config(config_path: Path) -> AppConfig:
    """读取 YAML 配置并构造 `AppConfig`。

    YAML 校验直接由 `AppConfig` 这套冻结 Pydantic dataclass 承担，
    调用方拿到的仍然是历史以来的 `AppConfig` 类型。
    """
    # safe_load 防止任意 Python 对象实例化；空文件返回 None 时退化为空 dict
    payload: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    try:
        return _check_loaded_config(_APP_CONFIG_ADAPTER.validate_python(payload))
    except ValidationError as exc:
        # Pydantic 在自定义校验器抛 ValueError 时会把 msg 包成 "Value error, <text>"。
        # 取第一条错误并剥去前缀，与旧实现的 ValueError 字节一致。
        first_error = exc.errors()[0]
        message = str(first_error["msg"]).removeprefix("Value error, ")
        raise ValueError(message) from exc
