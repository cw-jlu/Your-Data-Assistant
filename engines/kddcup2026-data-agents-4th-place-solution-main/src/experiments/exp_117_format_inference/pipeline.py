"""ReFoRCE-style 3-phase pipeline (= format + exploration + self_refine).

Top-level entry point. Faithfully mirrors ReFoRCE's run.py structure:

Phase 0: format inference  → estimate canonical answer header (= format_csv)
Phase 1: exploration       → 3-8 short SQLs, results become "few-shot examples"
Phase 2: self_refine       → iterative SQL gen with self-consistency, with
                              format_csv injected as the column constraint

Returns the final answer table or None if no consistent answer emerged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kobushi_core.benchmark.schema import AnswerTable, PublicTask
from kobushi_core.model import ModelAdapter

from experiments.exp_117_format_inference.format_inference import run_format_inference
from experiments.exp_117_format_inference.exploration import run_exploration_phase
from experiments.exp_117_format_inference.self_refine import (
    SelfRefineResult,
    run_self_refine,
)


@dataclass
class ReforcePipelineResult:
    ok: bool
    answer: AnswerTable | None
    format_csv: str
    exploration_meta: dict[str, Any]
    self_refine: SelfRefineResult | None
    failure_reason: str | None = None
    elapsed_seconds: float = 0.0


def run_pipeline(
    *,
    task: PublicTask,
    model: ModelAdapter,
    format_model: ModelAdapter | None = None,  # if None, reuses `model`
    max_iter: int = 5,
    max_exploration_queries: int = 8,
    early_stop_empty: int = 4,
) -> ReforcePipelineResult:
    import time
    t0 = time.time()

    # Phase 0: format inference
    fmt = run_format_inference(task=task, model=format_model or model)
    format_csv = fmt.get("format_csv", "")

    # Phase 1: exploration
    exp_meta = run_exploration_phase(
        task=task, model=model, max_queries=max_exploration_queries,
    )

    # Phase 2: self_refine, with the format header injected
    sr = run_self_refine(
        task=task, model=model,
        exploration_findings=exp_meta["findings"],
        format_csv=format_csv,
        max_iter=max_iter,
        early_stop_empty=early_stop_empty,
    )

    if not sr.ok:
        return ReforcePipelineResult(
            ok=False, answer=None,
            format_csv=format_csv,
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
        format_csv=format_csv,
        exploration_meta=exp_meta, self_refine=sr,
        elapsed_seconds=time.time() - t0,
    )
