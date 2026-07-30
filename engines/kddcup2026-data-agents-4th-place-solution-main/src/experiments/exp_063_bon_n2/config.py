from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv() -> None:
    """Load .env file from project root if it exists."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "qwen3.5-35b"
    api_base: str = ""
    api_key: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    max_steps: int = 16
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class BonConfig:
    temperatures: list[float]


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    bon: BonConfig = field(default_factory=lambda: BonConfig(temperatures=[0.0, 1.0]))
    run: RunConfig = field(default_factory=RunConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_app_config(config_path: Path) -> AppConfig:
    _load_dotenv()
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    bon_payload = payload.get("bon", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(
            dataset_payload.get("root_path")
            or os.environ.get("DATASET_ROOT_PATH"),
            dataset_defaults.root_path,
        ),
    )
    raw_extra_headers = agent_payload.get("extra_headers") or {}
    # Environment variables from .env override YAML values for credentials
    env_extra_headers = {}
    cf_client_id = os.environ.get("CF_ACCESS_CLIENT_ID", "")
    cf_client_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")
    if cf_client_id:
        env_extra_headers["CF-Access-Client-Id"] = cf_client_id
    if cf_client_secret:
        env_extra_headers["CF-Access-Client-Secret"] = cf_client_secret
    merged_headers = {str(k): str(v) for k, v in raw_extra_headers.items()}
    merged_headers.update(env_extra_headers)

    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=os.environ.get("AGENT_API_BASE")
        or str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=os.environ.get("AGENT_API_KEY")
        or str(agent_payload.get("api_key", agent_defaults.api_key)),
        extra_headers=merged_headers,
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
    )
    bon_defaults = BonConfig(temperatures=[0.0, 1.0])
    raw_temps = bon_payload.get("temperatures")
    bon_temperatures = [float(t) for t in raw_temps] if raw_temps is not None else bon_defaults.temperatures
    bon_config = BonConfig(temperatures=bon_temperatures)

    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(
            run_payload.get("output_dir") or os.environ.get("RUN_OUTPUT_DIR"),
            run_defaults.output_dir,
        ),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(
            run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)
        ),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, bon=bon_config, run=run_config)
