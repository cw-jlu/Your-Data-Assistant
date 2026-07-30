#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from kobushi_core.eval.csv_compare import EvaluationOptions, _evaluate_task  # noqa: E402


GOLD_ROOT = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "output"
RUN_ROOT = REPO / "artifacts" / "runs"
DEFAULT_RUNS = [
    "exp_150_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_010",
    "exp_151_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_004",
    "exp_152_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_final_sql_guard_003",
    "exp_153_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_final_sql_guard_audio_asr_002",
    "exp_153_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_final_sql_guard_audio_asr_003",
    "exp_153_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_final_sql_guard_audio_asr_004",
    "exp_154_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_audio_asr_003",
    "exp_154_answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard_audio_asr_004",
]


def _read_header(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0] if rows else []


def _run_names(args: argparse.Namespace) -> list[str]:
    names: list[str] = []
    if args.runs:
        names.extend([part.strip() for part in args.runs.split(",") if part.strip()])
    if args.run_glob:
        names.extend(sorted(path.name for path in RUN_ROOT.glob(args.run_glob) if path.is_dir()))
    if not names:
        names = list(DEFAULT_RUNS)
    seen = set()
    out = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _classify(ev) -> str:
    score = float(ev.official_score_lambda_0_5)
    if score >= 0.999:
        return "perfect"
    if ev.matched_cols == ev.gold_cols and ev.extra_cols > 0:
        return "extra_only_full_recall"
    if ev.matched_cols < ev.gold_cols and ev.extra_cols > 0:
        return "mixed_missing_and_extra"
    if ev.pred_cols > ev.gold_cols:
        return "too_many_cols_but_no_matched_extra"
    if ev.matched_cols < ev.gold_cols:
        return "missing_or_wrong_values"
    return "other_imperfect"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate whether a minimal-column advisor could help past Phase 2 predictions."
    )
    parser.add_argument("--runs", default=None, help="comma-separated run directory names")
    parser.add_argument("--run-glob", default=None, help="glob under artifacts/runs")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "artifacts" / "column_advisor_poc",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    task_rows: list[dict[str, object]] = []
    by_run: dict[str, Counter] = defaultdict(Counter)
    gain_by_run: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for run_name in _run_names(args):
        run_dir = RUN_ROOT / run_name
        for pred_path in sorted(run_dir.glob("task_*/prediction.csv")):
            task_id = pred_path.parent.name
            gold_path = GOLD_ROOT / task_id / "gold.csv"
            if not gold_path.is_file():
                continue
            try:
                ev = _evaluate_task(
                    task_id=task_id,
                    prediction_path=pred_path,
                    gold_path=gold_path,
                    options=EvaluationOptions(),
                )
            except Exception as exc:
                task_rows.append(
                    {
                        "run": run_name,
                        "task_id": task_id,
                        "classification": "eval_error",
                        "error": repr(exc),
                    }
                )
                by_run[run_name]["eval_error"] += 1
                continue

            classification = _classify(ev)
            score = float(ev.official_score_lambda_0_5)
            recall = float(ev.recall)
            drop_extra_gain = max(recall - score, 0.0)
            full_recall_extra_gain = max(1.0 - score, 0.0) if classification == "extra_only_full_recall" else 0.0
            by_run[run_name][classification] += 1
            by_run[run_name]["n"] += 1
            gain_by_run[run_name]["score"] += score
            gain_by_run[run_name]["drop_extra_gain"] += drop_extra_gain
            gain_by_run[run_name]["full_recall_extra_gain"] += full_recall_extra_gain

            task_rows.append(
                {
                    "run": run_name,
                    "task_id": task_id,
                    "classification": classification,
                    "score": score,
                    "matched_cols": ev.matched_cols,
                    "gold_cols": ev.gold_cols,
                    "pred_cols": ev.pred_cols,
                    "extra_cols": ev.extra_cols,
                    "recall": recall,
                    "extras_ratio": float(ev.extras_ratio),
                    "drop_extra_gain_upper_bound": drop_extra_gain,
                    "extra_only_gain": full_recall_extra_gain,
                    "gold_headers": " | ".join(_read_header(gold_path)),
                    "pred_headers": " | ".join(_read_header(pred_path)),
                    "error": "",
                }
            )

    summary_rows = []
    for run_name in _run_names(args):
        counts = by_run[run_name]
        n = counts["n"]
        if not n:
            continue
        gains = gain_by_run[run_name]
        summary_rows.append(
            {
                "run": run_name,
                "n": n,
                "mean_score": gains["score"] / n,
                "perfect": counts["perfect"],
                "extra_only_full_recall": counts["extra_only_full_recall"],
                "mixed_missing_and_extra": counts["mixed_missing_and_extra"],
                "too_many_cols_but_no_matched_extra": counts["too_many_cols_but_no_matched_extra"],
                "missing_or_wrong_values": counts["missing_or_wrong_values"],
                "other_imperfect": counts["other_imperfect"],
                "drop_extra_gain_upper_bound_mean": gains["drop_extra_gain"] / n,
                "extra_only_gain_mean": gains["full_recall_extra_gain"] / n,
            }
        )

    task_csv = args.out_dir / "past_run_column_opportunity_tasks.csv"
    summary_csv = args.out_dir / "past_run_column_opportunity_summary.csv"
    summary_json = args.out_dir / "past_run_column_opportunity_summary.json"

    task_fields = [
        "run",
        "task_id",
        "classification",
        "score",
        "matched_cols",
        "gold_cols",
        "pred_cols",
        "extra_cols",
        "recall",
        "extras_ratio",
        "drop_extra_gain_upper_bound",
        "extra_only_gain",
        "gold_headers",
        "pred_headers",
        "error",
    ]
    with task_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_fields)
        writer.writeheader()
        writer.writerows(task_rows)

    summary_fields = [
        "run",
        "n",
        "mean_score",
        "perfect",
        "extra_only_full_recall",
        "mixed_missing_and_extra",
        "too_many_cols_but_no_matched_extra",
        "missing_or_wrong_values",
        "other_imperfect",
        "drop_extra_gain_upper_bound_mean",
        "extra_only_gain_mean",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n")

    total = Counter()
    for row in task_rows:
        total[str(row["classification"])] += 1
    print("classification totals:")
    for key, value in total.most_common():
        print(f"  {key}: {value}")
    print(f"wrote {task_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_json}")


if __name__ == "__main__":
    main()
