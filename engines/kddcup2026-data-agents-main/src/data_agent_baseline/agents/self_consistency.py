"""Self-consistency wrapper around ``ReActAgent`` (G-2 / v6).

For hard and extreme difficulty tier the v5 50-task run showed ±2 perfect-task
variance between identical-config runs at temperature 0.0. The cause is the
qwen endpoint emitting subtly different chains-of-thought across runs even
when temperature is pinned, which sometimes shifts column choice or
filter scope. This wrapper executes the ReAct loop ``k`` times at a higher
temperature and majority-votes by ``column_signature`` over the normalized
answers — mirroring how the official scorer compares predictions.

Tie-break rule: when two buckets have the same size, the bucket whose
EARLIEST sample appears first wins. This keeps the run deterministic with
respect to the order of samples and lets us recover the same trace for
debugging.

The wrapper does not own the model adapter directly. Instead it accepts a
``model_factory`` callable so each sample gets its own adapter instance —
that re-initializes the JSON-mode probe state and queue position, which
matters when the first sample times out (see runner._FIRST_STEP_TRANSIENT_HINTS).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from data_agent_baseline.agents.model import ModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.scoring.normalize import column_signature
from data_agent_baseline.tools.registry import ToolRegistry


SignatureKey = frozenset[tuple[frozenset[tuple[str, int]], int]]


def _answer_signature(table: AnswerTable | None) -> SignatureKey | None:
    """Hashable bucket key for voting.

    Each column produces its own ``column_signature`` (multiset of
    normalized values, ignoring column NAME and row ORDER). We then count
    how many columns share each signature and freeze the result. This
    matches the official scorer's column-multiset semantics exactly,
    including the case where two predicted columns happen to carry the
    same content (each contributes once to the count).
    """
    if table is None:
        return None
    column_sigs: list[frozenset[tuple[str, int]]] = []
    for col_idx in range(len(table.columns)):
        values = [
            str(row[col_idx]) if col_idx < len(row) else ""
            for row in table.rows
        ]
        column_sigs.append(column_signature(values))
    return frozenset(Counter(column_sigs).items())


@dataclass(frozen=True, slots=True)
class SelfConsistencyConfig:
    k: int = 3
    react_config: ReActAgentConfig | None = None


class SelfConsistencyAgent:
    """Run ``ReActAgent`` ``k`` times and majority-vote on the answer.

    Parameters
    ----------
    model_factory:
        Zero-arg callable that returns a fresh ``ModelAdapter`` per
        sample. The runner wires this to ``OpenAIModelAdapter`` at the
        higher self-consistency temperature; tests pass a factory that
        yields ``ScriptedModelAdapter`` instances.
    tools:
        Shared ``ToolRegistry`` — each sample shares filesystem state but
        the persistent kernel is reset by ``cleanup_task`` at the end of
        every ``ReActAgent.run`` (already handled in ReActAgent's finally).
    config:
        ``ReActAgentConfig`` forwarded to each sample's agent.
    k:
        Number of samples; ``k <= 1`` collapses to a single ReAct run.
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], ModelAdapter] | None = None,
        model_factory_with_index: Callable[[int], ModelAdapter] | None = None,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        k: int = 3,
        policy_hints: tuple[str, ...] | None = None,
        preferred_tools: tuple[str, ...] | None = None,
        avoid_tools: tuple[str, ...] | None = None,
        max_steps_multiplier: float = 1.0,
        total_budget_seconds: float | None = None,
    ) -> None:
        if model_factory is None and model_factory_with_index is None:
            raise ValueError("Must provide model_factory or model_factory_with_index.")
        # Index-aware factory wins when both supplied (per-sample seed wiring).
        # Legacy callers (tests) keep the no-arg model_factory shape working.
        if model_factory_with_index is not None:
            self._model_factory_with_index: Callable[[int], ModelAdapter] = model_factory_with_index
        else:
            assert model_factory is not None  # narrowed by branch above
            captured = model_factory
            self._model_factory_with_index = lambda _idx: captured()
        self._tools = tools
        self._config = config
        self._k = max(1, int(k))
        # v7 memory-layer overrides — forwarded to each sample's ReActAgent.
        self._policy_hints = tuple(policy_hints or ())
        self._preferred_tools = tuple(preferred_tools or ())
        self._avoid_tools = tuple(avoid_tools or ())
        self._max_steps_multiplier = float(max_steps_multiplier or 1.0)
        # v4 fix #1: total wall-clock budget across all k samples. When set
        # the run() method allocates each sample
        #   per_sample = max(remaining_budget / remaining_samples, 0)
        # and passes the corresponding absolute deadline to ReActAgent.run.
        # None disables budgeting (legacy behavior: each sample runs full).
        self._total_budget_seconds = (
            float(total_budget_seconds) if total_budget_seconds is not None else None
        )

    @property
    def k(self) -> int:
        return self._k

    def _run_one_sample(
        self,
        task: PublicTask,
        *,
        deadline: float | None = None,
        sample_idx: int = 0,
    ) -> AgentRunResult:
        adapter = self._model_factory_with_index(sample_idx)
        agent = ReActAgent(
            model=adapter,
            tools=self._tools,
            config=self._config,
            policy_hints=self._policy_hints,
            preferred_tools=self._preferred_tools,
            avoid_tools=self._avoid_tools,
            max_steps_multiplier=self._max_steps_multiplier,
        )
        try:
            return agent.run(task, deadline=deadline)
        except Exception as exc:  # noqa: BLE001 — preserve other samples
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                normalized_answer=None,
                steps=[],
                failure_reason=f"Sample raised: {exc}",
            )

    def run(self, task: PublicTask) -> AgentRunResult:
        if self._k <= 1:
            return self._run_one_sample(task, sample_idx=0)

        samples: list[AgentRunResult] = []
        started = perf_counter()
        for idx in range(self._k):
            sample_deadline: float | None = None
            if self._total_budget_seconds is not None:
                elapsed = perf_counter() - started
                remaining = self._total_budget_seconds - elapsed
                # If the budget is exhausted, abort the remaining samples.
                # We already have at least one result from earlier samples
                # OR this is the first sample with zero budget — voter will
                # fall through to ``samples[0]`` and return None-answer.
                if remaining <= 0:
                    break
                remaining_samples = self._k - idx
                per_sample = remaining / remaining_samples
                sample_deadline = perf_counter() + per_sample
            samples.append(self._run_one_sample(task, deadline=sample_deadline, sample_idx=idx))

        if not samples:
            # Total budget was zero/negative before any sample could start.
            return AgentRunResult(
                task_id=task.task_id,
                answer=None,
                normalized_answer=None,
                steps=[],
                failure_reason="Self-consistency total budget exhausted before first sample.",
            )

        # Bucket non-None answers by signature, preserving sample order.
        buckets: dict[SignatureKey, list[int]] = {}
        for idx, sample in enumerate(samples):
            sig = _answer_signature(sample.normalized_answer)
            if sig is None:
                continue
            buckets.setdefault(sig, []).append(idx)

        if not buckets:
            return samples[0]

        # max with key=(size, -earliest_index): bigger bucket wins;
        # on tie, the bucket containing the EARLIEST sample wins.
        winning_indices = max(buckets.values(), key=lambda idxs: (len(idxs), -min(idxs)))
        return samples[winning_indices[0]]
