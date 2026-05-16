"""
此模块负责加载和解析应用配置，包括数据集路径、Agent 设置及运行参数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import os
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "qwen3.5-35b-a3b"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0
    max_tokens: int | None = 8192
    rag_top_k: int = 5  # PageRAG/GraphRAG 检索返回的最大分块数


@dataclass(frozen=True, slots=True)
class WikiConfig:
    enabled: bool = True
    wiki_root: Path = field(default_factory=lambda: PROJECT_ROOT / "wiki")
    retrieval_top_k: int = 5
    auto_ingest: bool = True


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
    run: RunConfig = field(default_factory=RunConfig)
    wiki: WikiConfig = field(default_factory=WikiConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()
    wiki_defaults = WikiConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})
    wiki_payload = payload.get("wiki", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        model=os.environ.get("MODEL_NAME") or str(agent_payload.get("model", agent_defaults.model)),
        api_base=os.environ.get("MODEL_API_URL") or str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=os.environ.get("MODEL_API_KEY") or str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        max_tokens=agent_payload.get("max_tokens", agent_defaults.max_tokens),
        rag_top_k=int(agent_payload.get("rag_top_k", agent_defaults.rag_top_k)),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )

    wiki_root_raw = wiki_payload.get("wiki_root")
    wiki_root = _path_value(str(wiki_root_raw), wiki_defaults.wiki_root) if wiki_root_raw else wiki_defaults.wiki_root

    wiki_config = WikiConfig(
        enabled=bool(wiki_payload.get("enabled", wiki_defaults.enabled)),
        wiki_root=wiki_root,
        retrieval_top_k=int(wiki_payload.get("retrieval_top_k", wiki_defaults.retrieval_top_k)),
        auto_ingest=bool(wiki_payload.get("auto_ingest", wiki_defaults.auto_ingest)),
    )

    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config, wiki=wiki_config)
