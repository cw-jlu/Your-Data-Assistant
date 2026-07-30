"""Column-ablation diagnostic — propose drops that lift Score across all λ.

The official Score formula is ``Recall − λ·(ExtraCols/PredictedCols)``.
When the agent over-emits a column (one with no signature match in gold),
dropping it lowers ExtraCols *without* lowering Matched, so Score rises.

We surface drop candidates per task as a **diagnostic** — never modifying
``prediction.csv``. The strict gate: every λ in ``{0.05, 0.10, 0.20}``
must agree the drop is favourable. The competition λ is undisclosed, and
optimizing for our local guess (0.10) over-fits.

Use via ``mock_scorer --ablate`` or ``propose_column_ablation`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.scoring.mock_scorer import score_one

DEFAULT_ABLATION_LAMBDAS: tuple[float, ...] = (0.05, 0.10, 0.20)


@dataclass(frozen=True, slots=True)
class AblationCandidate:
    task_id: str
    column_index: int
    column_name: str
    score_keep: dict[float, float]
    score_drop: dict[float, float]
    delta_min: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "column_index": self.column_index,
            "column_name": self.column_name,
            "score_keep": {f"{k:.2f}": round(v, 4) for k, v in self.score_keep.items()},
            "score_drop": {f"{k:.2f}": round(v, 4) for k, v in self.score_drop.items()},
            "delta_min": round(self.delta_min, 4),
        }


def _drop_column(table: AnswerTable, col_index: int) -> AnswerTable:
    columns = list(table.columns)
    rows = list(table.rows)
    new_columns = columns[:col_index] + columns[col_index + 1:]
    new_rows = [row[:col_index] + row[col_index + 1:] for row in rows]
    return AnswerTable(columns=new_columns, rows=new_rows)


def _score_at(prediction: AnswerTable, gold: AnswerTable, lam: float) -> float:
    score_value, *_ = score_one(prediction=prediction, gold=gold, lambda_value=lam)
    return score_value


def propose_column_ablation_for_task(
    *,
    task_id: str,
    prediction: AnswerTable,
    gold: AnswerTable,
    lambdas: Iterable[float] = DEFAULT_ABLATION_LAMBDAS,
    min_columns: int = 2,
) -> AblationCandidate | None:
    """Return the best single-column drop for this task, or None.

    The candidate is the column whose removal yields a strictly positive
    Score delta for every λ in ``lambdas``. Among such columns the one
    with the largest *minimum* delta wins (worst-case improvement).

    Returns ``None`` when the prediction has fewer than ``min_columns``
    columns or no drop universally helps.
    """
    lambdas_t = tuple(lambdas)
    if len(prediction.columns) < min_columns:
        return None

    score_keep: dict[float, float] = {lam: _score_at(prediction, gold, lam) for lam in lambdas_t}

    best: AblationCandidate | None = None
    for col_idx in range(len(prediction.columns)):
        dropped = _drop_column(prediction, col_idx)
        if not dropped.columns:
            continue
        score_drop: dict[float, float] = {lam: _score_at(dropped, gold, lam) for lam in lambdas_t}
        deltas = [score_drop[lam] - score_keep[lam] for lam in lambdas_t]
        if not all(d > 0 for d in deltas):
            continue
        delta_min = min(deltas)
        if best is None or delta_min > best.delta_min:
            best = AblationCandidate(
                task_id=task_id,
                column_index=col_idx,
                column_name=prediction.columns[col_idx],
                score_keep=score_keep,
                score_drop=score_drop,
                delta_min=delta_min,
            )
    return best


def propose_column_ablation(
    *,
    predictions: dict[str, AnswerTable],
    gold: dict[str, AnswerTable],
    lambdas: Iterable[float] = DEFAULT_ABLATION_LAMBDAS,
    min_columns: int = 2,
) -> list[AblationCandidate]:
    """Run ``propose_column_ablation_for_task`` over every (pred, gold) pair."""
    lambdas_t = tuple(lambdas)
    candidates: list[AblationCandidate] = []
    for task_id, pred in sorted(predictions.items()):
        gold_table = gold.get(task_id)
        if gold_table is None:
            continue
        cand = propose_column_ablation_for_task(
            task_id=task_id,
            prediction=pred,
            gold=gold_table,
            lambdas=lambdas_t,
            min_columns=min_columns,
        )
        if cand is not None:
            candidates.append(cand)
    return candidates
