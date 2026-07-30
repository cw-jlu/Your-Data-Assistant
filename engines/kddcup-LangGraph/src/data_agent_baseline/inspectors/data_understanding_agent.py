from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import ContextView, PublicTask, TaskAssets
from data_agent_baseline.config import DataInspectorConfig
from data_agent_baseline.inspectors.semantic_catalog import build_lightweight_catalog, build_semantic_catalog


class DataUnderstandingAgent:
    def __init__(self, config: DataInspectorConfig) -> None:
        self.config = config

    def explore_data_globally(
        self,
        *,
        context_dir: Path,
        task_id: str = "",
        context_view: ContextView | None = None,
    ) -> tuple[str, dict[str, Any]]:
        catalog = build_semantic_catalog(
            PublicTask(
                record=type("TaskRecord", (), {"task_id": task_id, "difficulty": "", "question": ""})(),
                assets=TaskAssets(
                    task_dir=context_dir,
                    context_dir=context_dir,
                    context_view=context_view,
                ),
            ),
            budget=self.config.sample_budget,
            semantic_view_config=self.config.semantic_views,
        )
        lightweight_catalog = build_lightweight_catalog(catalog)
        return json.dumps(lightweight_catalog, ensure_ascii=False, indent=2), catalog
