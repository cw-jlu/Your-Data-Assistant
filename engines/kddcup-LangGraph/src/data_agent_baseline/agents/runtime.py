from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    node: str
    assistant_message: str | None
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    ok: bool
    model_request: dict[str, Any] | None = None
    model_response: dict[str, Any] | None = None
    started_at: str | None = None
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None
    inspector: dict[str, Any] | None = None
    global_data_profile: str | None = None
    ambiguity_analysis: dict[str, Any] | None = None
    semantic_ledger: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.failure_reason,
            "succeeded": self.succeeded,
            "inspector": self.inspector,
            "global_data_profile": self.global_data_profile,
            "ambiguity_analysis": self.ambiguity_analysis,
            "semantic_ledger": self.semantic_ledger,
        }
