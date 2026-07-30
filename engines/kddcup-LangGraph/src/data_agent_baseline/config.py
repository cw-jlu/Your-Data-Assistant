from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# 项目根目录用于解析默认数据路径和相对配置路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# 默认公开数据集目录。
def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "input"


# 默认运行产物目录。
def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


# 数据集相关配置。
@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


# Agent 相关配置，包括模型信息和单任务最大模型轮数。
@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    model_env: str | None = None
    api_base: str = "https://api.openai.com/v1"
    api_base_env: str | None = None
    api_key: str = ""
    api_key_env: str | None = None
    max_steps: int = 16
    temperature: float = 0.0
    model_request_timeout_seconds: int | None = 1800
    max_tokens: int | None = 8192
    enable_data_inspector: bool = False
    enable_answer_validator: bool = True
    validation_retry_limit: int = 2
    enable_process_validator: bool = False
    enable_ambiguity_analysis: bool = False
    strip_reasoning_history: bool = False
    reasoning_history_limit: int | None = None
    prompt_version: int = 1
    compress_used_image_messages: bool = True
    compressed_image_note_chars: int = 600

    def __post_init__(self) -> None:
        if self.validation_retry_limit < 0:
            raise ValueError("agent.validation_retry_limit must be non-negative.")
        if self.compressed_image_note_chars < 0:
            raise ValueError("agent.compressed_image_note_chars must be non-negative.")


@dataclass(frozen=True, slots=True)
class DataInspectorSampleBudget:
    catalog_top_distinct_values: int = 50
    max_doc_tokens: int = 500


@dataclass(frozen=True, slots=True)
class DataInspectorSemanticViewConfig:
    enabled: bool = True
    min_confidence: float = 0.95
    min_distinct_match_ratio: float = 0.90
    strict_distinct_match_ratio: float = 0.95
    min_target_uniqueness_ratio: float = 1.0
    max_views: int = 30
    max_dimension_fields_per_view: int = 12
    max_dimensions_per_view: int = 2
    min_payload_fields: int = 1
    allow_one_to_one_enrichment: bool = False

    def __post_init__(self) -> None:
        if self.min_confidence < 0 or self.min_confidence > 1:
            raise ValueError("data_inspector.semantic_views.min_confidence must be between 0 and 1.")
        if self.min_distinct_match_ratio < 0 or self.min_distinct_match_ratio > 1:
            raise ValueError(
                "data_inspector.semantic_views.min_distinct_match_ratio must be between 0 and 1."
            )
        if self.strict_distinct_match_ratio < 0 or self.strict_distinct_match_ratio > 1:
            raise ValueError(
                "data_inspector.semantic_views.strict_distinct_match_ratio must be between 0 and 1."
            )
        if self.min_target_uniqueness_ratio < 0 or self.min_target_uniqueness_ratio > 1:
            raise ValueError(
                "data_inspector.semantic_views.min_target_uniqueness_ratio must be between 0 and 1."
            )
        if self.max_views < 0:
            raise ValueError("data_inspector.semantic_views.max_views must be non-negative.")
        if self.max_dimension_fields_per_view < 0:
            raise ValueError(
                "data_inspector.semantic_views.max_dimension_fields_per_view must be non-negative."
            )
        if self.max_dimensions_per_view < 0:
            raise ValueError("data_inspector.semantic_views.max_dimensions_per_view must be non-negative.")
        if self.min_payload_fields < 0:
            raise ValueError("data_inspector.semantic_views.min_payload_fields must be non-negative.")


@dataclass(frozen=True, slots=True)
class DataInspectorConfig:
    sample_budget: DataInspectorSampleBudget = field(default_factory=DataInspectorSampleBudget)
    semantic_views: DataInspectorSemanticViewConfig = field(
        default_factory=DataInspectorSemanticViewConfig
    )


@dataclass(frozen=True, slots=True)
class ProcessValidatorConfig:
    checkpoint_model_interval: int = 10
    retry_limit: int = 5
    recent_step_limit: int = 8
    evidence_max_items: int = 12
    evidence_max_str_tokens: int = 300
    evidence_max_list_items: int = 5

    def __post_init__(self) -> None:
        if self.checkpoint_model_interval <= 0:
            raise ValueError("process_validator.checkpoint_model_interval must be a positive integer.")
        if self.retry_limit < 0:
            raise ValueError("process_validator.retry_limit must be a non-negative integer.")
        if self.recent_step_limit <= 0:
            raise ValueError("process_validator.recent_step_limit must be a positive integer.")
        if self.evidence_max_items <= 0:
            raise ValueError("process_validator.evidence_max_items must be a positive integer.")
        if self.evidence_max_str_tokens <= 0:
            raise ValueError("process_validator.evidence_max_str_tokens must be a positive integer.")
        if self.evidence_max_list_items <= 0:
            raise ValueError("process_validator.evidence_max_list_items must be a positive integer.")


@dataclass(frozen=True, slots=True)
class VideoPreprocessingConfig:
    enabled: bool = True
    sample_fps: float = 2.0
    diff_threshold: float = 0.025
    pixel_delta: int = 25
    min_stable_duration: float = 1.0
    resize_width: int = 320
    jpg_quality: int = 95
    max_attached_frames: int = 16
    asr_model: str = "medium"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_cpu_threads: int = 4
    asr_num_workers: int = 8
    asr_stage: str = "dynamic-plus-ui"
    asr_language_detector_model: str = "tiny"
    asr_language_threshold: float = 0.5
    asr_prompt_min_terms: int = 1
    asr_prompt_max_terms: int = 10
    asr_ui_terms: tuple[str, ...] = (
        "字段",
        "批次",
        "配置",
        "加载",
        "面板",
        "看板",
        "卡片",
        "队列",
        "筛选",
        "核对",
        "列",
        "路径",
    )

    def __post_init__(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError("video_preprocessing.sample_fps must be positive.")
        if self.diff_threshold < 0:
            raise ValueError("video_preprocessing.diff_threshold must be non-negative.")
        if self.pixel_delta < 0:
            raise ValueError("video_preprocessing.pixel_delta must be non-negative.")
        if self.min_stable_duration < 0:
            raise ValueError("video_preprocessing.min_stable_duration must be non-negative.")
        if self.resize_width <= 0:
            raise ValueError("video_preprocessing.resize_width must be positive.")
        if self.jpg_quality < 1 or self.jpg_quality > 100:
            raise ValueError("video_preprocessing.jpg_quality must be between 1 and 100.")
        if self.max_attached_frames < 0:
            raise ValueError("video_preprocessing.max_attached_frames must be non-negative.")
        if self.asr_cpu_threads < 1:
            raise ValueError("video_preprocessing.asr_cpu_threads must be positive.")
        if self.asr_num_workers < 1:
            raise ValueError("video_preprocessing.asr_num_workers must be positive.")
        if self.asr_stage not in {
            "medium-baseline",
            "tiny-route",
            "dynamic-prompt",
            "dynamic-plus-ui",
        }:
            raise ValueError("video_preprocessing.asr_stage is unsupported.")
        if not 0 <= self.asr_language_threshold <= 1:
            raise ValueError(
                "video_preprocessing.asr_language_threshold must be between 0 and 1."
            )
        if self.asr_prompt_min_terms < 1:
            raise ValueError("video_preprocessing.asr_prompt_min_terms must be positive.")
        if self.asr_prompt_max_terms < self.asr_prompt_min_terms:
            raise ValueError(
                "video_preprocessing.asr_prompt_max_terms must be at least "
                "asr_prompt_min_terms."
            )
        if not self.asr_ui_terms or any(not str(term).strip() for term in self.asr_ui_terms):
            raise ValueError("video_preprocessing.asr_ui_terms must contain non-empty terms.")


@dataclass(frozen=True, slots=True)
class StructuredDocLLMConfig:
    """Sampling settings used only for extract_structured_doc model calls."""

    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_tokens: int = 16384
    repair_temperature: float = 0.3

    def __post_init__(self) -> None:
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("tool.structured_doc.llm.temperature must be between 0 and 2.")
        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError("tool.structured_doc.llm.top_p must be greater than 0 and at most 1.")
        if self.repetition_penalty <= 0:
            raise ValueError("tool.structured_doc.llm.repetition_penalty must be positive.")
        if self.max_tokens < 1:
            raise ValueError("tool.structured_doc.llm.max_tokens must be positive.")
        if self.repair_temperature < 0 or self.repair_temperature > 2:
            raise ValueError(
                "tool.structured_doc.llm.repair_temperature must be between 0 and 2."
            )


@dataclass(frozen=True, slots=True)
class StructuredDocToolConfig:
    min_chunk_lines: int = 25
    max_chunk_lines: int = 40
    max_selected_lines_for_llm_extraction: int = 400
    default_max_model_calls: int = 20
    hard_max_model_calls: int = 20
    inspect_doc_structure_max_model_calls: int = 3
    llm: StructuredDocLLMConfig = field(default_factory=StructuredDocLLMConfig)

    def __post_init__(self) -> None:
        for field_name in (
            "min_chunk_lines",
            "max_chunk_lines",
            "max_selected_lines_for_llm_extraction",
            "default_max_model_calls",
            "hard_max_model_calls",
            "inspect_doc_structure_max_model_calls",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"tool.structured_doc.{field_name} must be a positive integer.")
        if self.min_chunk_lines > self.max_chunk_lines:
            raise ValueError(
                "tool.structured_doc.min_chunk_lines must be less than or equal to "
                "tool.structured_doc.max_chunk_lines."
            )
        if self.default_max_model_calls > self.hard_max_model_calls:
            raise ValueError(
                "tool.structured_doc.default_max_model_calls must be less than or equal to "
                "tool.structured_doc.hard_max_model_calls."
            )
        if self.inspect_doc_structure_max_model_calls > self.hard_max_model_calls:
            raise ValueError(
                "tool.structured_doc.inspect_doc_structure_max_model_calls must be less "
                "than or equal to tool.structured_doc.hard_max_model_calls."
            )


@dataclass(frozen=True, slots=True)
class ToolConfig:
    max_output_tokens: int = 10000
    max_list_items: int = 200
    structured_doc: StructuredDocToolConfig = field(default_factory=StructuredDocToolConfig)


def _optional_string_value(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    normalized = str(raw_value).strip()
    return normalized or None


def _bool_value(raw_value: object, default_value: bool) -> bool:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Expected a boolean value, got: {raw_value!r}")


def _float_value(raw_value: object, default_value: float) -> float:
    if raw_value is None:
        return default_value
    return float(raw_value)


def _optional_non_negative_int_value(raw_value: object, default_value: int | None, *, field_name: str) -> int | None:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, str) and raw_value.strip().lower() in {"", "none", "null"}:
        return None
    if isinstance(raw_value, bool):
        raise ValueError(f"{field_name} must be null or a non-negative integer.")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, str):
        value = int(raw_value.strip())
    else:
        raise ValueError(f"{field_name} must be null or a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be null or a non-negative integer.")
    return value


def _non_negative_int_value(raw_value: object, default_value: int, *, field_name: str) -> int:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, str) and raw_value.strip():
        value = int(raw_value.strip())
    else:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _positive_int_value(raw_value: object, default_value: int, *, field_name: str) -> int:
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, str) and raw_value.strip():
        value = int(raw_value.strip())
    else:
        raise ValueError(f"{field_name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _string_list_value(raw_value: object, *, field_name: str) -> tuple[str, ...] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, list):
        raise ValueError(f"Expected `{field_name}` to be a YAML list.")

    normalized_values: list[str] = []
    for raw_item in raw_value:
        item = str(raw_item).strip()
        if item:
            normalized_values.append(item)

    if not normalized_values:
        return None

    # 保序去重，避免重复任务被重复执行。
    return tuple(dict.fromkeys(normalized_values))


def _dotenv_value(dotenv_path: Path, env_var_name: str) -> str | None:
    if not dotenv_path.exists():
        return None

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        name, raw_value = line.split("=", 1)
        if name.strip() != env_var_name:
            continue

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value.strip()

    return None


def _env_value(env_var_name: str) -> str | None:
    value = os.environ.get(env_var_name, "").strip()
    return value or None


def _resolve_env_backed_string(
    *,
    raw_value: object,
    raw_env: object,
    default_value: str,
    field_name: str,
) -> tuple[str, str | None]:
    env_var_name = _optional_string_value(raw_env)
    if env_var_name is not None:
        value = _env_value(env_var_name)
        if value is None:
            raise ValueError(f"Missing environment variable `{env_var_name}` for `{field_name}`.")
        return value, env_var_name
    return str(raw_value if raw_value is not None else default_value).strip(), None


def _resolve_api_key(raw_api_key: object, raw_api_key_env: object, default_api_key: str) -> tuple[str, str | None]:
    api_key = str(raw_api_key if raw_api_key is not None else default_api_key).strip()
    api_key_env = _optional_string_value(raw_api_key_env)
    if api_key or api_key_env is None:
        return api_key, api_key_env

    env_api_key = _env_value(api_key_env)
    if env_api_key is not None:
        return env_api_key, api_key_env

    dotenv_api_key = _dotenv_value(PROJECT_ROOT / ".env", api_key_env)
    return (dotenv_api_key or "").strip(), api_key_env


# 运行时配置，包括输出目录、并发度、任务超时和提交态软时限。
@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    log_dir: Path | None = None
    output_layout: str = "run_dir"
    run_id: str | None = None
    max_workers: int = 4
    extract_structured_doc_max_workers: int = 2
    task_timeout_seconds: int = 600
    extract_structured_doc_timeout_bonus_seconds: int = 0
    task_ids: tuple[str, ...] | None = None


# 顶层应用配置，把 dataset / agent / run 三组配置聚合在一起。
@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data_inspector: DataInspectorConfig = field(default_factory=DataInspectorConfig)
    process_validator: ProcessValidatorConfig = field(default_factory=ProcessValidatorConfig)
    video_preprocessing: VideoPreprocessingConfig = field(default_factory=VideoPreprocessingConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    run: RunConfig = field(default_factory=RunConfig)


# 把 YAML 中的路径字段解析成 Path；相对路径默认相对于项目根目录。
def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    raw_text = str(raw_value).strip()
    if not raw_text:
        return default_value
    candidate = Path(raw_text)
    if candidate.is_absolute() or raw_text.startswith("/"):
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _output_layout_value(raw_value: object, default_value: str) -> str:
    output_layout = str(raw_value if raw_value is not None else default_value).strip().lower()
    if output_layout not in {"run_dir", "flat"}:
        raise ValueError("run.output_layout must be either `run_dir` or `flat`.")
    return output_layout



def _structured_doc_tool_config_value(raw_value: object | None) -> StructuredDocToolConfig:
    defaults = StructuredDocToolConfig()
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError("tool.structured_doc must be a YAML object.")
    llm_payload = raw_value.get("llm", {})
    if not isinstance(llm_payload, dict):
        raise ValueError("tool.structured_doc.llm must be a YAML object.")
    llm_defaults = defaults.llm
    return StructuredDocToolConfig(
        min_chunk_lines=_positive_int_value(
            raw_value.get("min_chunk_lines"),
            defaults.min_chunk_lines,
            field_name="tool.structured_doc.min_chunk_lines",
        ),
        max_chunk_lines=_positive_int_value(
            raw_value.get("max_chunk_lines"),
            defaults.max_chunk_lines,
            field_name="tool.structured_doc.max_chunk_lines",
        ),
        max_selected_lines_for_llm_extraction=_positive_int_value(
            raw_value.get("max_selected_lines_for_llm_extraction"),
            defaults.max_selected_lines_for_llm_extraction,
            field_name="tool.structured_doc.max_selected_lines_for_llm_extraction",
        ),
        default_max_model_calls=_positive_int_value(
            raw_value.get("default_max_model_calls"),
            defaults.default_max_model_calls,
            field_name="tool.structured_doc.default_max_model_calls",
        ),
        hard_max_model_calls=_positive_int_value(
            raw_value.get("hard_max_model_calls"),
            defaults.hard_max_model_calls,
            field_name="tool.structured_doc.hard_max_model_calls",
        ),
        inspect_doc_structure_max_model_calls=_positive_int_value(
            raw_value.get("inspect_doc_structure_max_model_calls"),
            defaults.inspect_doc_structure_max_model_calls,
            field_name="tool.structured_doc.inspect_doc_structure_max_model_calls",
        ),
        llm=StructuredDocLLMConfig(
            temperature=_float_value(llm_payload.get("temperature"), llm_defaults.temperature),
            top_p=_float_value(llm_payload.get("top_p"), llm_defaults.top_p),
            repetition_penalty=_float_value(
                llm_payload.get("repetition_penalty"),
                llm_defaults.repetition_penalty,
            ),
            max_tokens=_positive_int_value(
                llm_payload.get("max_tokens"),
                llm_defaults.max_tokens,
                field_name="tool.structured_doc.llm.max_tokens",
            ),
            repair_temperature=_float_value(
                llm_payload.get("repair_temperature"),
                llm_defaults.repair_temperature,
            ),
        ),
    )


def _tool_config_value(raw_value: object | None) -> ToolConfig:
    defaults = ToolConfig()
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError("tool must be a YAML object.")
    return ToolConfig(
        max_output_tokens=int(raw_value.get("max_output_tokens", defaults.max_output_tokens)),
        max_list_items=int(raw_value.get("max_list_items", defaults.max_list_items)),
        structured_doc=_structured_doc_tool_config_value(raw_value.get("structured_doc")),
    )


def _data_inspector_sample_budget_value(raw_value: object | None) -> DataInspectorSampleBudget:
    defaults = DataInspectorSampleBudget()
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError("data_inspector.sample_budget must be a YAML object.")
    return DataInspectorSampleBudget(
        catalog_top_distinct_values=int(raw_value.get("catalog_top_distinct_values", defaults.catalog_top_distinct_values)),
        max_doc_tokens=int(raw_value.get("max_doc_tokens", defaults.max_doc_tokens)),
    )


def _semantic_view_config_value(raw_value: object | None) -> DataInspectorSemanticViewConfig:
    defaults = DataInspectorSemanticViewConfig()
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError("data_inspector.semantic_views must be a YAML object.")
    return DataInspectorSemanticViewConfig(
        enabled=_bool_value(raw_value.get("enabled"), defaults.enabled),
        min_confidence=_float_value(raw_value.get("min_confidence"), defaults.min_confidence),
        min_distinct_match_ratio=_float_value(
            raw_value.get("min_distinct_match_ratio"),
            defaults.min_distinct_match_ratio,
        ),
        strict_distinct_match_ratio=_float_value(
            raw_value.get("strict_distinct_match_ratio"),
            defaults.strict_distinct_match_ratio,
        ),
        min_target_uniqueness_ratio=_float_value(
            raw_value.get("min_target_uniqueness_ratio"),
            defaults.min_target_uniqueness_ratio,
        ),
        max_views=int(raw_value.get("max_views", defaults.max_views)),
        max_dimension_fields_per_view=int(
            raw_value.get(
                "max_dimension_fields_per_view",
                defaults.max_dimension_fields_per_view,
            )
        ),
        max_dimensions_per_view=int(
            raw_value.get("max_dimensions_per_view", defaults.max_dimensions_per_view)
        ),
        min_payload_fields=int(raw_value.get("min_payload_fields", defaults.min_payload_fields)),
        allow_one_to_one_enrichment=_bool_value(
            raw_value.get("allow_one_to_one_enrichment"),
            defaults.allow_one_to_one_enrichment,
        ),
    )


def _data_inspector_config_value(raw_value: object | None) -> DataInspectorConfig:
    defaults = DataInspectorConfig()
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError("data_inspector must be a YAML object.")
    return DataInspectorConfig(
        sample_budget=_data_inspector_sample_budget_value(raw_value.get("sample_budget")),
        semantic_views=_semantic_view_config_value(raw_value.get("semantic_views")),
    )


def _process_validator_config_value(raw_value: object | None) -> ProcessValidatorConfig:
    defaults = ProcessValidatorConfig()
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError("process_validator must be a YAML object.")
    return ProcessValidatorConfig(
        checkpoint_model_interval=int(
            raw_value.get("checkpoint_model_interval", defaults.checkpoint_model_interval)
        ),
        retry_limit=int(raw_value.get("retry_limit", defaults.retry_limit)),
        recent_step_limit=int(raw_value.get("recent_step_limit", defaults.recent_step_limit)),
        evidence_max_items=int(
            raw_value.get("evidence_max_items", defaults.evidence_max_items)
        ),
        evidence_max_str_tokens=int(
            raw_value.get("evidence_max_str_tokens", defaults.evidence_max_str_tokens)
        ),
        evidence_max_list_items=int(
            raw_value.get("evidence_max_list_items", defaults.evidence_max_list_items)
        ),
    )


def _video_preprocessing_config_value(raw_value: object | None) -> VideoPreprocessingConfig:
    defaults = VideoPreprocessingConfig()
    if raw_value is None:
        return defaults
    if not isinstance(raw_value, dict):
        raise ValueError("video_preprocessing must be a YAML object.")
    return VideoPreprocessingConfig(
        enabled=_bool_value(raw_value.get("enabled"), defaults.enabled),
        sample_fps=_float_value(raw_value.get("sample_fps"), defaults.sample_fps),
        diff_threshold=_float_value(raw_value.get("diff_threshold"), defaults.diff_threshold),
        pixel_delta=int(raw_value.get("pixel_delta", defaults.pixel_delta)),
        min_stable_duration=_float_value(
            raw_value.get("min_stable_duration"),
            defaults.min_stable_duration,
        ),
        resize_width=int(raw_value.get("resize_width", defaults.resize_width)),
        jpg_quality=int(raw_value.get("jpg_quality", defaults.jpg_quality)),
        max_attached_frames=int(
            raw_value.get("max_attached_frames", defaults.max_attached_frames)
        ),
        asr_model=str(raw_value.get("asr_model", defaults.asr_model)).strip()
        or defaults.asr_model,
        asr_device=str(raw_value.get("asr_device", defaults.asr_device)).strip()
        or defaults.asr_device,
        asr_compute_type=str(
            raw_value.get("asr_compute_type", defaults.asr_compute_type)
        ).strip()
        or defaults.asr_compute_type,
        asr_cpu_threads=int(raw_value.get("asr_cpu_threads", defaults.asr_cpu_threads)),
        asr_num_workers=int(raw_value.get("asr_num_workers", defaults.asr_num_workers)),
        asr_stage=str(raw_value.get("asr_stage", defaults.asr_stage)).strip()
        or defaults.asr_stage,
        asr_language_detector_model=str(
            raw_value.get(
                "asr_language_detector_model",
                defaults.asr_language_detector_model,
            )
        ).strip()
        or defaults.asr_language_detector_model,
        asr_language_threshold=_float_value(
            raw_value.get("asr_language_threshold"),
            defaults.asr_language_threshold,
        ),
        asr_prompt_min_terms=int(
            raw_value.get("asr_prompt_min_terms", defaults.asr_prompt_min_terms)
        ),
        asr_prompt_max_terms=int(
            raw_value.get("asr_prompt_max_terms", defaults.asr_prompt_max_terms)
        ),
        asr_ui_terms=tuple(
            str(term).strip()
            for term in raw_value.get("asr_ui_terms", defaults.asr_ui_terms)
        ),
    )


# 从 YAML 配置文件加载应用配置，并对缺省值和相对路径做统一处理。
def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    data_inspector_payload = payload.get("data_inspector", {})
    process_validator_payload = payload.get("process_validator", {})
    video_preprocessing_payload = payload.get("video_preprocessing", {})
    tool_payload = payload.get("tool", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    model, model_env = _resolve_env_backed_string(
        raw_value=agent_payload.get("model", agent_defaults.model),
        raw_env=agent_payload.get("model_env"),
        default_value=agent_defaults.model,
        field_name="agent.model",
    )
    api_base, api_base_env = _resolve_env_backed_string(
        raw_value=agent_payload.get("api_base", agent_defaults.api_base),
        raw_env=agent_payload.get("api_base_env"),
        default_value=agent_defaults.api_base,
        field_name="agent.api_base",
    )
    api_key, api_key_env = _resolve_api_key(
        agent_payload.get("api_key", agent_defaults.api_key),
        agent_payload.get("api_key_env"),
        agent_defaults.api_key,
    )
    agent_config = AgentConfig(
        model=model,
        model_env=model_env,
        api_base=api_base,
        api_base_env=api_base_env,
        api_key=api_key,
        api_key_env=api_key_env,
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=_float_value(agent_payload.get("temperature"), agent_defaults.temperature),
        model_request_timeout_seconds=_optional_non_negative_int_value(
            agent_payload.get("model_request_timeout_seconds"),
            agent_defaults.model_request_timeout_seconds,
            field_name="agent.model_request_timeout_seconds",
        ),
        max_tokens=_optional_non_negative_int_value(
            agent_payload.get("max_tokens"),
            agent_defaults.max_tokens,
            field_name="agent.max_tokens",
        ),
        enable_data_inspector=_bool_value(
            agent_payload.get("enable_data_inspector"),
            agent_defaults.enable_data_inspector,
        ),
        enable_answer_validator=_bool_value(
            agent_payload.get("enable_answer_validator"),
            agent_defaults.enable_answer_validator,
        ),
        validation_retry_limit=_non_negative_int_value(
            agent_payload.get("validation_retry_limit"),
            agent_defaults.validation_retry_limit,
            field_name="agent.validation_retry_limit",
        ),
        enable_process_validator=_bool_value(
            agent_payload.get("enable_process_validator"),
            agent_defaults.enable_process_validator,
        ),
        enable_ambiguity_analysis=_bool_value(
            agent_payload.get("enable_ambiguity_analysis"),
            agent_defaults.enable_ambiguity_analysis,
        ),
        strip_reasoning_history=_bool_value(
            agent_payload.get("strip_reasoning_history"),
            agent_defaults.strip_reasoning_history,
        ),
        reasoning_history_limit=_optional_non_negative_int_value(
            agent_payload.get("reasoning_history_limit"),
            agent_defaults.reasoning_history_limit,
            field_name="agent.reasoning_history_limit",
        ),
        prompt_version=int(agent_payload.get("prompt_version", agent_defaults.prompt_version)),
        compress_used_image_messages=_bool_value(
            agent_payload.get("compress_used_image_messages"),
            agent_defaults.compress_used_image_messages,
        ),
        compressed_image_note_chars=int(
            agent_payload.get(
                "compressed_image_note_chars",
                agent_defaults.compressed_image_note_chars,
            )
        ),
    )
    data_inspector_config = _data_inspector_config_value(data_inspector_payload)
    process_validator_config = _process_validator_config_value(process_validator_payload)
    video_preprocessing_config = _video_preprocessing_config_value(video_preprocessing_payload)
    tool_config = _tool_config_value(tool_payload)
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        # 空字符串视为未指定 run_id，交由 runner 自动生成。
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        log_dir=(
            _path_value(run_payload.get("log_dir"), run_defaults.output_dir)
            if run_payload.get("log_dir") is not None
            else None
        ),
        output_layout=_output_layout_value(run_payload.get("output_layout"), run_defaults.output_layout),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        extract_structured_doc_max_workers=_positive_int_value(
            run_payload.get("extract_structured_doc_max_workers"),
            run_defaults.extract_structured_doc_max_workers,
            field_name="run.extract_structured_doc_max_workers",
        ),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
        extract_structured_doc_timeout_bonus_seconds=_non_negative_int_value(
            run_payload.get("extract_structured_doc_timeout_bonus_seconds"),
            run_defaults.extract_structured_doc_timeout_bonus_seconds,
            field_name="run.extract_structured_doc_timeout_bonus_seconds",
        ),
        task_ids=_string_list_value(run_payload.get("task_ids"), field_name="run.task_ids"),
    )
    return AppConfig(
        dataset=dataset_config,
        agent=agent_config,
        data_inspector=data_inspector_config,
        process_validator=process_validator_config,
        video_preprocessing=video_preprocessing_config,
        tool=tool_config,
        run=run_config,
    )
