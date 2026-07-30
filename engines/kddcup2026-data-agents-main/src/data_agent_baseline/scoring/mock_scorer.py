"""Local mock of the official DataAgent-Bench scorer.

Mirrors the scoring rule from https://dataagent.top/rules:

    Score = Recall − λ · (ExtraCols / PredictedCols)

Where:
- columns are matched by their normalized multi-set value signature
  (sorted-value semantics, ignoring column names and row order),
- normalization is done by `scoring/normalize.normalize_answer_table`,
- Recall = MatchedCols / GoldCols,
- ExtraCols = max(PredictedCols − MatchedCols, 0).

λ defaults to 0.10 because the rules page does not publish the official
value. The `--lambda-values` flag lets you sweep multiple λ to find a
robust answer; column-ablation in Phase 3 uses this.

Per-task score is in [-λ, 1]; we clip to ≥ 0 because the leaderboard is
non-negative (an answer worse than empty just gets 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.scoring.normalize import (
    column_signature,
    normalize_answer_table,
)

DEFAULT_LAMBDA = 0.10
LAMBDA_ENV_VAR = "DABENCH_LAMBDA"


@dataclass(frozen=True, slots=True)
class TaskScore:
    task_id: str
    difficulty: str | None
    recall: float
    score: float
    matched_columns: int
    gold_columns: int
    predicted_columns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "recall": round(self.recall, 4),
            "score": round(self.score, 4),
            "matched_columns": self.matched_columns,
            "gold_columns": self.gold_columns,
            "predicted_columns": self.predicted_columns,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkScore:
    lambda_value: float
    task_scores: list[TaskScore]
    missing_predictions: list[str]
    missing_gold: list[str]

    @property
    def mean_score(self) -> float:
        if not self.task_scores:
            return 0.0
        return statistics.mean(t.score for t in self.task_scores)

    @property
    def mean_recall(self) -> float:
        if not self.task_scores:
            return 0.0
        return statistics.mean(t.recall for t in self.task_scores)

    def by_difficulty(self) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[TaskScore]] = {}
        for entry in self.task_scores:
            key = entry.difficulty or "unknown"
            grouped.setdefault(key, []).append(entry)
        return {
            difficulty: {
                "task_count": float(len(entries)),
                "mean_score": statistics.mean(t.score for t in entries),
                "mean_recall": statistics.mean(t.recall for t in entries),
            }
            for difficulty, entries in grouped.items()
        }


def resolve_lambda(explicit: float | None = None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(LAMBDA_ENV_VAR)
    if raw:
        return float(raw)
    return DEFAULT_LAMBDA


def _read_csv_table(path: Path) -> AnswerTable:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return AnswerTable(columns=[], rows=[])
    columns, *data_rows = rows
    width = len(columns)
    typed_rows = [list(row) + [""] * (width - len(row)) if len(row) < width else list(row[:width]) for row in data_rows]
    return AnswerTable(columns=columns, rows=typed_rows)


def _column_signatures(table: AnswerTable) -> list[frozenset[tuple[str, int]]]:
    if not table.columns:
        return []
    columns_data: list[list[str]] = [[] for _ in table.columns]
    for row in table.rows:
        for index in range(len(table.columns)):
            value = row[index] if index < len(row) else ""
            columns_data[index].append(str(value))
    return [column_signature(values) for values in columns_data]


def _name_pair_signature(
    table: AnswerTable, idx_a: int, idx_b: int
) -> frozenset[tuple[str, int]]:
    """Row-wise space-joined signature for two normalized columns.

    Empty values on either side collapse to the non-empty side (so a row
    with one missing piece still contributes the present token); fully
    empty rows produce ``""`` so they cannot accidentally match anything.
    """
    joined: list[str] = []
    for row in table.rows:
        a = str(row[idx_a]).strip() if idx_a < len(row) else ""
        b = str(row[idx_b]).strip() if idx_b < len(row) else ""
        if not a and not b:
            joined.append("")
        elif not a:
            joined.append(b)
        elif not b:
            joined.append(a)
        else:
            joined.append(f"{a} {b}")
    return column_signature(joined)


def _pair_signature_matches(
    table: AnswerTable, idx_a: int, idx_b: int, target: frozenset[tuple[str, int]]
) -> bool:
    """Compare a pair-joined signature against a target signature.

    Tries both orders (a-then-b, b-then-a) since name-equivalence under
    rules §10 admits 'FirstName LastName' but does not specify which
    column is which.
    """
    return (
        _name_pair_signature(table, idx_a, idx_b) == target
        or _name_pair_signature(table, idx_b, idx_a) == target
    )


def score_one(
    *,
    prediction: AnswerTable,
    gold: AnswerTable,
    lambda_value: float = DEFAULT_LAMBDA,
) -> tuple[float, float, int, int, int]:
    """Returns (score, recall, matched, gold_cols, predicted_cols).

    Matching algorithm (greedy, three phases):

    1. Direct one-to-one: each gold column tries to find a pred column
       with the same multiset signature (post-normalization).
    2. Pair → single: an unmatched gold column tries to match the
       row-wise join of two unused pred columns. This satisfies
       rules §10 'FirstName + LastName accepted as FirstName LastName'.
       One gold column claimed; two pred columns consumed.
    3. Single → pair: an unmatched pred column tries to match the join
       of two unused gold columns. Two gold columns claimed; one pred
       column consumed.

    ExtraCols is pred_cols minus the number of pred columns actually
    used in any matching phase.
    """
    normalized_pred, _ = normalize_answer_table(prediction)
    normalized_gold, _ = normalize_answer_table(gold)
    pred_signatures = _column_signatures(normalized_pred)
    gold_signatures = _column_signatures(normalized_gold)
    if not gold_signatures:
        return 0.0, 0.0, 0, 0, len(pred_signatures)

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matched = 0

    # Phase 1 — direct one-to-one
    for gold_idx, gold_sig in enumerate(gold_signatures):
        if gold_idx in used_gold:
            continue
        for pred_idx, pred_sig in enumerate(pred_signatures):
            if pred_idx in used_pred:
                continue
            if pred_sig == gold_sig:
                used_pred.add(pred_idx)
                used_gold.add(gold_idx)
                matched += 1
                break

    # Phase 2 — gold single ⇄ pred pair join (e.g. gold has full_name,
    # pred has first_name + last_name).
    pred_count = len(pred_signatures)
    for gold_idx, gold_sig in enumerate(gold_signatures):
        if gold_idx in used_gold:
            continue
        matched_pair = False
        for i in range(pred_count):
            if i in used_pred:
                continue
            for j in range(i + 1, pred_count):
                if j in used_pred:
                    continue
                if _pair_signature_matches(normalized_pred, i, j, gold_sig):
                    used_pred.add(i)
                    used_pred.add(j)
                    used_gold.add(gold_idx)
                    matched += 1
                    matched_pair = True
                    break
            if matched_pair:
                break

    # Phase 3 — pred single ⇄ gold pair join (e.g. pred has full_name,
    # gold has first_name + last_name). Both gold columns count as
    # matched, so recall reflects the rules-§10 equivalence.
    gold_count = len(gold_signatures)
    for pred_idx, pred_sig in enumerate(pred_signatures):
        if pred_idx in used_pred:
            continue
        matched_pair = False
        for i in range(gold_count):
            if i in used_gold:
                continue
            for j in range(i + 1, gold_count):
                if j in used_gold:
                    continue
                if _pair_signature_matches(normalized_gold, i, j, pred_sig):
                    used_gold.add(i)
                    used_gold.add(j)
                    used_pred.add(pred_idx)
                    matched += 2
                    matched_pair = True
                    break
            if matched_pair:
                break

    gold_cols = gold_count
    pred_cols = pred_count
    recall = matched / gold_cols if gold_cols > 0 else 0.0
    extra = max(pred_cols - len(used_pred), 0)
    penalty = lambda_value * (extra / pred_cols) if pred_cols > 0 else 0.0
    score = max(0.0, recall - penalty)
    return score, recall, matched, gold_cols, pred_cols


def _read_difficulty_index(predictions_root: Path) -> dict[str, str]:
    """Pull difficulty per task_id from any trace.json in the predictions tree."""
    index: dict[str, str] = {}
    for task_dir in predictions_root.iterdir() if predictions_root.is_dir() else []:
        if not task_dir.is_dir():
            continue
        trace_path = task_dir / "trace.json"
        if not trace_path.exists():
            continue
        try:
            payload = json.loads(trace_path.read_text())
        except json.JSONDecodeError:
            continue
        difficulty = payload.get("difficulty")
        if isinstance(difficulty, str):
            index[task_dir.name] = difficulty
    return index


def _load_task_difficulty_from_input(input_root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    if not input_root.is_dir():
        return index
    for task_dir in input_root.iterdir():
        if not task_dir.is_dir():
            continue
        task_json = task_dir / "task.json"
        if not task_json.exists():
            continue
        try:
            payload = json.loads(task_json.read_text())
        except json.JSONDecodeError:
            continue
        difficulty = payload.get("difficulty")
        if isinstance(difficulty, str):
            index[task_dir.name] = difficulty
    return index


def score_benchmark(
    *,
    predictions_root: Path,
    gold_root: Path,
    input_root: Path | None = None,
    task_filter: Iterable[str] | None = None,
    lambda_value: float = DEFAULT_LAMBDA,
) -> BenchmarkScore:
    if not gold_root.is_dir():
        raise FileNotFoundError(f"Gold root not found: {gold_root}")
    gold_task_ids = sorted(
        path.name for path in gold_root.iterdir() if path.is_dir() and (path / "gold.csv").exists()
    )
    if task_filter is not None:
        wanted = set(task_filter)
        gold_task_ids = [t for t in gold_task_ids if t in wanted]

    difficulty_index: dict[str, str] = {}
    if input_root is not None:
        difficulty_index.update(_load_task_difficulty_from_input(input_root))
    difficulty_index.update(_read_difficulty_index(predictions_root))

    task_scores: list[TaskScore] = []
    missing_predictions: list[str] = []
    missing_gold: list[str] = []

    for task_id in gold_task_ids:
        gold_path = gold_root / task_id / "gold.csv"
        prediction_path = predictions_root / task_id / "prediction.csv"
        if not prediction_path.exists():
            missing_predictions.append(task_id)
            continue
        if not gold_path.exists():
            missing_gold.append(task_id)
            continue

        prediction = _read_csv_table(prediction_path)
        gold = _read_csv_table(gold_path)
        score_value, recall, matched, gold_cols, pred_cols = score_one(
            prediction=prediction,
            gold=gold,
            lambda_value=lambda_value,
        )
        task_scores.append(
            TaskScore(
                task_id=task_id,
                difficulty=difficulty_index.get(task_id),
                recall=recall,
                score=score_value,
                matched_columns=matched,
                gold_columns=gold_cols,
                predicted_columns=pred_cols,
            )
        )

    return BenchmarkScore(
        lambda_value=lambda_value,
        task_scores=task_scores,
        missing_predictions=sorted(set(missing_predictions)),
        missing_gold=sorted(set(missing_gold)),
    )


def render_text_summary(result: BenchmarkScore) -> str:
    lines = [
        f"λ = {result.lambda_value:.2f}",
        f"tasks scored: {len(result.task_scores)}",
        f"mean Score:   {result.mean_score:.4f}",
        f"mean Recall:  {result.mean_recall:.4f}",
    ]
    if result.missing_predictions:
        lines.append(f"missing predictions ({len(result.missing_predictions)}): {result.missing_predictions[:8]}")
    if result.missing_gold:
        lines.append(f"missing gold ({len(result.missing_gold)}): {result.missing_gold[:8]}")
    grouped = result.by_difficulty()
    if grouped:
        lines.append("by difficulty:")
        for difficulty in sorted(grouped):
            stats = grouped[difficulty]
            lines.append(
                f"  {difficulty:<10} n={int(stats['task_count']):>4}  "
                f"score={stats['mean_score']:.4f}  recall={stats['mean_recall']:.4f}"
            )
    return "\n".join(lines)


def _read_all_tables(
    predictions_root: Path,
    gold_root: Path,
    task_filter: Iterable[str] | None,
) -> tuple[dict[str, AnswerTable], dict[str, AnswerTable]]:
    """Load matching prediction.csv and gold.csv pairs as AnswerTable dicts."""
    if not gold_root.is_dir():
        return {}, {}
    gold_task_ids = sorted(
        path.name for path in gold_root.iterdir() if path.is_dir() and (path / "gold.csv").exists()
    )
    if task_filter is not None:
        wanted = set(task_filter)
        gold_task_ids = [t for t in gold_task_ids if t in wanted]
    predictions: dict[str, AnswerTable] = {}
    gold_tables: dict[str, AnswerTable] = {}
    for task_id in gold_task_ids:
        pred_path = predictions_root / task_id / "prediction.csv"
        gold_path = gold_root / task_id / "gold.csv"
        if not pred_path.exists() or not gold_path.exists():
            continue
        predictions[task_id] = _read_csv_table(pred_path)
        gold_tables[task_id] = _read_csv_table(gold_path)
    return predictions, gold_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predictions against gold.")
    parser.add_argument("--predictions", type=Path, required=True, help="Directory with task_<id>/prediction.csv")
    parser.add_argument("--gold", type=Path, required=True, help="Directory with task_<id>/gold.csv")
    parser.add_argument("--input", type=Path, default=None, help="Dataset input root for difficulty lookup.")
    parser.add_argument(
        "--task-filter",
        type=Path,
        default=None,
        help="File listing one task_id per line; only those are scored.",
    )
    parser.add_argument("--lambda-value", type=float, default=None, help="Override λ; falls back to env DABENCH_LAMBDA, then 0.10.")
    parser.add_argument("--lambda-values", type=float, nargs="+", default=None, help="Score with multiple λ values for sensitivity analysis.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON summary instead of text.")
    parser.add_argument(
        "--ablate",
        action="store_true",
        help=(
            "Diagnostic: propose single-column drops that would lift Score across all λ. "
            "Does not modify predictions; uses --lambda-values (default {0.05,0.10,0.20})."
        ),
    )
    args = parser.parse_args()

    task_filter: list[str] | None = None
    if args.task_filter is not None:
        task_filter = [line.strip() for line in args.task_filter.read_text().splitlines() if line.strip()]

    lambdas = args.lambda_values if args.lambda_values else [resolve_lambda(args.lambda_value)]
    summaries = []
    for lam in lambdas:
        result = score_benchmark(
            predictions_root=args.predictions,
            gold_root=args.gold,
            input_root=args.input,
            task_filter=task_filter,
            lambda_value=lam,
        )
        if args.json:
            summaries.append(
                {
                    "lambda": lam,
                    "mean_score": result.mean_score,
                    "mean_recall": result.mean_recall,
                    "task_count": len(result.task_scores),
                    "by_difficulty": result.by_difficulty(),
                    "missing_predictions": result.missing_predictions,
                    "missing_gold": result.missing_gold,
                    "tasks": [t.to_dict() for t in result.task_scores],
                }
            )
        else:
            print(render_text_summary(result))
            print("---")

    if args.json:
        print(json.dumps(summaries, indent=2))

    if args.ablate:
        from data_agent_baseline.scoring.column_ablation import (
            DEFAULT_ABLATION_LAMBDAS,
            propose_column_ablation,
        )

        ablation_lambdas = lambdas if len(lambdas) >= 2 else list(DEFAULT_ABLATION_LAMBDAS)
        predictions, gold_tables = _read_all_tables(args.predictions, args.gold, task_filter)
        candidates = propose_column_ablation(
            predictions=predictions,
            gold=gold_tables,
            lambdas=ablation_lambdas,
        )
        if args.json:
            print(json.dumps({"ablation_candidates": [c.to_dict() for c in candidates]}, indent=2))
        else:
            print()
            print(f"=== Column ablation (λ={ablation_lambdas}) ===")
            if not candidates:
                print("  no drop is universally favourable across all λ values")
            else:
                for c in candidates:
                    print(
                        f"  {c.task_id}: drop col {c.column_index} "
                        f"({c.column_name!r}) → +{c.delta_min:.4f} (min Δ across λ)"
                    )


if __name__ == "__main__":
    main()
