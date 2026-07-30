"""Local scorer for the KDD Cup 2026 Data Agents leaderboard rule.

Produces ``scores.json`` from a predictions tree and a gold tree. Each answer
column is treated as one unordered column-value vector ("column multiset");
matching is greedy one-to-one across columns with row order ignored.

Per-task math::

    recall  = matched_columns / gold_columns
    penalty = lambda * (extra_columns / predicted_columns)   # 0 when pred has no columns
    score   = max(0, recall - penalty)

The benchmark total is the simple mean of per-task ``score`` over every task
that has a ``gold.csv`` (missing predictions count as 0).

Layout expected::

    <pred_root>/<task_id>/prediction.csv
    <gold_root>/<task_id>/gold.csv

Pure-version typical invocation::

    uv run python scripts/score_predictions.py \\
        --pred artifacts/runs/<run_id> \\
        --gold data/public/output \\
        --lambda 0.1 \\
        --output artifacts/runs/<run_id>/scores.json \\
        --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATETIME_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
)

DATE_ONLY_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
)

NUMBER_WITH_COMMAS = re.compile(r"-?[\d,]+(\.\d+)?")

NULL_TOKENS = frozenset({"", "null", "none", "nan", "nat", "<na>"})

_TWO_PLACES = Decimal("0.01")


def _canonical_number(text: str) -> str | None:
    """Parse a numeric string per official rule §6.5.

    Rule: ``Decimal``-parsed and quantized to 2 decimal places with
    ``ROUND_HALF_UP``. So ``18 → "18.00"``, ``4200000 → "4200000.00"``,
    ``0.005 → "0.01"`` (half-up, **not** banker's rounding).

    Returns ``None`` for non-finite or unparseable input so the caller can fall
    through to the date / raw-string paths.
    """
    try:
        d = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None
    quantized = d.quantize(_TWO_PLACES, rounding=ROUND_HALF_UP)
    if quantized == 0:
        # quantize keeps the sign on tiny negatives (e.g. -0.001 → -0.00).
        # Collapse signed zero to "0.00" so +0 and -0 hash the same.
        return "0.00"
    return format(quantized, "f")


def normalize_cell(value: Any, *, strict_commas: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer, Decimal, np.floating, float)):
        # Round-trip through str() so a Python float like 0.005 (binary
        # 0.00499999999999999989...) canonicalizes via its short repr "0.005"
        # — Decimal(0.005) directly would carry the noisy binary expansion
        # through quantize and yield "0.00" instead of the official "0.01".
        canonical = _canonical_number(str(value))
        if canonical is not None:
            return canonical
        # inf / nan: fall through to the raw-string path below

    if isinstance(value, (pd.Timestamp, datetime)):
        if getattr(value, "tzinfo", None) is not None:
            return value.astimezone(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return value.isoformat()

    text = str(value).strip().replace("\r\n", "\n")
    candidate = (
        text
        if strict_commas
        else (text.replace(",", "") if NUMBER_WITH_COMMAS.fullmatch(text) else text)
    )
    canonical = _canonical_number(candidate)
    if canonical is not None:
        return canonical

    # datetime first (more specific) — "2024-03-01T00:00:00" must not be
    # consumed by the date-only branch below.
    for fmt in DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            return parsed.astimezone(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return parsed.isoformat()

    for fmt in DATE_ONLY_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed.strftime("%Y-%m-%d")

    return text


def column_signature(values: list[Any], *, strict_commas: bool = False) -> str:
    normalized = sorted(normalize_cell(v, strict_commas=strict_commas) for v in values)
    joined = "\x1f".join(normalized)
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class TaskScore:
    status: str
    score: float = 0.0
    recall: float = 0.0
    penalty: float = 0.0
    gold_columns: int = 0
    predicted_columns: int = 0
    matched_columns: int = 0
    extra_columns: int = 0
    mismatched_columns: list[dict[str, Any]] = field(default_factory=list)
    extra_column_fingerprints: list[str] = field(default_factory=list)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def score_task(
    pred_csv: Path,
    gold_csv: Path,
    lambda_val: float,
    *,
    strict_commas: bool = False,
) -> TaskScore:
    if not pred_csv.exists():
        return TaskScore(status="missing_prediction")

    try:
        pred_df = _read_csv(pred_csv)
        gold_df = _read_csv(gold_csv)
    except Exception as exc:
        return TaskScore(status=f"read_error:{type(exc).__name__}")

    pred_sigs = Counter(
        column_signature(list(pred_df[c]), strict_commas=strict_commas) for c in pred_df.columns
    )
    gold_sigs = Counter(
        column_signature(list(gold_df[c]), strict_commas=strict_commas) for c in gold_df.columns
    )

    matched = sum(min(pred_sigs[s], gold_sigs[s]) for s in gold_sigs)
    gold_cols = sum(gold_sigs.values())
    pred_cols = sum(pred_sigs.values())
    extra = max(0, pred_cols - matched)

    recall = matched / gold_cols if gold_cols else 0.0
    penalty = lambda_val * (extra / pred_cols) if pred_cols else 0.0
    score = max(0.0, recall - penalty)

    mismatched = []
    for sig, gcount in gold_sigs.items():
        if pred_sigs.get(sig, 0) < gcount:
            mismatched.append({"gold_fingerprint": sig, "reason": "values differ or missing"})

    extra_fps: list[str] = []
    for sig, pcount in pred_sigs.items():
        surplus = pcount - gold_sigs.get(sig, 0)
        if surplus > 0:
            extra_fps.extend([sig] * surplus)

    if recall == 1.0 and extra == 0:
        status = "scored"
    elif matched > 0:
        status = "partial_match"
    else:
        status = "no_match"

    return TaskScore(
        status=status,
        score=score,
        recall=recall,
        penalty=penalty,
        gold_columns=gold_cols,
        predicted_columns=pred_cols,
        matched_columns=matched,
        extra_columns=extra,
        mismatched_columns=mismatched,
        extra_column_fingerprints=extra_fps,
    )


def _task_difficulty(input_root: Path | None, task_id: str) -> str:
    if input_root is None:
        return "unknown"
    meta_path = input_root / task_id / "task.json"
    if not meta_path.is_file():
        return "unknown"
    try:
        return str(json.loads(meta_path.read_text()).get("difficulty", "unknown"))
    except Exception:
        return "unknown"


def aggregate(per_task: dict[str, TaskScore], task_meta: dict[str, str]) -> dict[str, Any]:
    if not per_task:
        return {
            "total_score": 0.0,
            "mean_recall": 0.0,
            "tasks_scored_nonzero": 0,
            "tasks_missing": 0,
            "by_difficulty": {},
        }
    total = sum(s.score for s in per_task.values()) / len(per_task)
    mean_recall = sum(s.recall for s in per_task.values()) / len(per_task)
    by_diff: dict[str, list[float]] = defaultdict(list)
    for tid, score in per_task.items():
        by_diff[task_meta.get(tid, "unknown")].append(score.score)
    return {
        "total_score": total,
        "mean_recall": mean_recall,
        "tasks_scored_nonzero": sum(1 for s in per_task.values() if s.score > 0),
        "tasks_missing": sum(1 for s in per_task.values() if s.status == "missing_prediction"),
        "by_difficulty": {
            d: {"count": len(scores), "mean_score": sum(scores) / len(scores)}
            for d, scores in sorted(by_diff.items())
        },
    }


def check_regression(current: dict[str, Any], previous_path: Path) -> list[str]:
    previous = json.loads(previous_path.read_text())
    regressions: list[str] = []
    for tid, cur in current["per_task"].items():
        if tid not in previous.get("per_task", {}):
            continue
        prev_score = float(previous["per_task"][tid].get("score", 0))
        cur_score = float(cur.get("score", 0))
        if prev_score > 0 and cur_score == 0:
            regressions.append(f"{tid}: {prev_score:.2f} → 0 (SEVERE)")
        elif cur_score < prev_score - 0.2:
            regressions.append(f"{tid}: {prev_score:.2f} → {cur_score:.2f} (warn)")
    return regressions


def _discover_task_ids(pred_dir: Path, gold_dir: Path) -> list[str]:
    pred_ids = {p.name for p in pred_dir.iterdir() if p.is_dir()} if pred_dir.exists() else set()
    gold_ids = {p.name for p in gold_dir.iterdir() if p.is_dir()} if gold_dir.exists() else set()
    return sorted(pred_ids | gold_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local scorer for KDD Cup 2026 Data Agents.")
    parser.add_argument(
        "--pred", required=True, type=Path, help="Predictions dir (task_<id>/prediction.csv)."
    )
    parser.add_argument(
        "--gold", required=True, type=Path, help="Ground-truth dir (task_<id>/gold.csv)."
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_val",
        type=float,
        default=0.1,
        help="Penalty weight λ (default 0.1).",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Input dir for difficulty lookup (defaults to sibling 'input/' of --gold).",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--strict-commas",
        action="store_true",
        help="Do NOT strip thousands commas before numeric coercion.",
    )
    parser.add_argument(
        "--fail-on-regression",
        type=Path,
        default=None,
        help="Previous scores.json to compare against.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Continue even if regressions detected."
    )
    args = parser.parse_args(argv)

    gold_dir: Path = args.gold
    pred_dir: Path = args.pred
    input_root = args.input_root or (gold_dir.parent / "input")

    if not gold_dir.is_dir():
        print(f"ERROR: --gold dir missing: {gold_dir}", file=sys.stderr)
        return 2
    if not pred_dir.is_dir():
        # Fail fast — silent miss would emit a "0.0 across the board" scores.json
        # that looks legitimate, masking a typo'd path.
        print(f"ERROR: --pred dir missing: {pred_dir}", file=sys.stderr)
        return 2

    task_ids = _discover_task_ids(pred_dir, gold_dir)
    if not task_ids:
        print("ERROR: no task_* dirs found under --pred or --gold", file=sys.stderr)
        return 2

    per_task: dict[str, TaskScore] = {}
    task_meta: dict[str, str] = {}
    for task_id in task_ids:
        gold_csv = gold_dir / task_id / "gold.csv"
        pred_csv = pred_dir / task_id / "prediction.csv"
        if not gold_csv.exists():
            continue
        score = score_task(pred_csv, gold_csv, args.lambda_val, strict_commas=args.strict_commas)
        per_task[task_id] = score
        task_meta[task_id] = _task_difficulty(input_root, task_id)

        if args.verbose:
            print(
                f"{task_id:>12}  diff={task_meta[task_id]:<8}  status={score.status:<18}  "
                f"recall={score.recall:.3f}  penalty={score.penalty:.3f}  score={score.score:.3f}",
                file=sys.stderr,
            )

    result = {
        "meta": {
            "pred_dir": str(pred_dir),
            "gold_dir": str(gold_dir),
            "lambda": args.lambda_val,
            "strict_commas": args.strict_commas,
            "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_tasks_gold": sum(1 for t in task_ids if (gold_dir / t / "gold.csv").exists()),
            "n_tasks_pred": sum(1 for t in task_ids if (pred_dir / t / "prediction.csv").exists()),
        },
        "per_task": {tid: asdict(score) for tid, score in per_task.items()},
        "aggregate": aggregate(per_task, task_meta),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    agg = result["aggregate"]
    print(
        f"total_score={agg['total_score']:.4f}  "
        f"mean_recall={agg['mean_recall']:.4f}  "
        f"scored_nonzero={agg['tasks_scored_nonzero']}/{len(per_task)}  "
        f"missing={agg['tasks_missing']}"
    )

    if args.fail_on_regression is not None:
        if not args.fail_on_regression.is_file():
            # Fail fast — silently skipping a typo'd reference path would let
            # genuine regressions slip through pre-submit checks unnoticed.
            print(
                f"ERROR: --fail-on-regression file missing: {args.fail_on_regression}",
                file=sys.stderr,
            )
            return 2
        regressions = check_regression(result, args.fail_on_regression)
        if regressions:
            print("REGRESSIONS:", file=sys.stderr)
            for line in regressions:
                print(f"  {line}", file=sys.stderr)
            if not args.force:
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
