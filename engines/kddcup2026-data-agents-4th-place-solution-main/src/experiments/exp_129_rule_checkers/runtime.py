from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from kobushi_core.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRuntimeState:
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None
    # Accumulated per-output-token top-logprobs across the attempt.
    # Each element is a dict {token: logprob}, one per generated token.
    # Empty unless ReActAgent was constructed with collect_logprobs=True.
    logprobs_buffer: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None
    preamble: str | None = None
    # AIMO3-style mean entropy across all output tokens this attempt produced.
    # float('inf') when no logprobs were collected (= disabled).
    mean_entropy: float = float("inf")
    n_logprob_tokens: int = 0

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
            "preamble": self.preamble,
            "mean_entropy": self.mean_entropy,
            "n_logprob_tokens": self.n_logprob_tokens,
        }
