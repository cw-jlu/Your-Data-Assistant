"""ReFoRCE-style 2-phase pipeline (= exploration + self_refine).

Top-level entry point. Replaces the ReAct-style multi-tool loop with the
fixed 2-phase flow from ReFoRCE's official agent.

Phase 1: exploration → 3-8 short SQLs → results become "few-shot examples"
Phase 2: self_refine → iterative SQL generation until self-consistent

Returns the final answer table or None if no consistent answer emerged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.model import ModelAdapter

from experiments.exp_116_reforce_loop.exploration import run_exploration_phase
from experiments.exp_116_reforce_loop.self_refine import (
    SelfRefineResult,
    run_self_refine,
)


@dataclass
class ReforcePipelineResult:
    ok: bool
    answer: AnswerTable | None
    exploration_meta: dict[str, Any]
    self_refine: SelfRefineResult | None
    failure_reason: str | None = None
    elapsed_seconds: float = 0.0


def run_pipeline(
    *,
    task: PublicTask,
    model: ModelAdapter,
    max_iter: int = 5,
    max_exploration_queries: int = 8,
    early_stop_empty: int = 4,
) -> ReforcePipelineResult:
    import time
    t0 = time.time()

    # Phase 1: exploration
    exp_meta = run_exploration_phase(
        task=task, model=model, max_queries=max_exploration_queries,
    )

    # Phase 2: self_refine
    sr = run_self_refine(
        task=task, model=model,
        exploration_findings=exp_meta["findings"],
        max_iter=max_iter,
        early_stop_empty=early_stop_empty,
    )

    if not sr.ok:
        return ReforcePipelineResult(
            ok=False, answer=None,
            exploration_meta=exp_meta, self_refine=sr,
            failure_reason=f"self_refine: {sr.commit_reason}",
            elapsed_seconds=time.time() - t0,
        )

    answer = AnswerTable(
        columns=list(sr.columns),
        rows=[[("" if v is None else str(v)) for v in r] for r in sr.rows],
    )
    return ReforcePipelineResult(
        ok=True, answer=answer,
        exploration_meta=exp_meta, self_refine=sr,
        elapsed_seconds=time.time() - t0,
    )
