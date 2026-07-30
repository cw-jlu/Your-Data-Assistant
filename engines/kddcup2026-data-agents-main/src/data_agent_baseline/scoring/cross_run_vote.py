"""Cross-run prediction voter (H-1).

Take N independent benchmark runs of the SAME code on the SAME task set,
majority-vote each task's prediction by column-signature, and emit a
single voted output tree that mock_scorer can consume.

Why this exists
---------------

v6 forensics demonstrated that temperature=0 still produces ±3-5 perfect-
task variance per run on easy/medium tier. SelfConsistencyAgent already
suppresses this for hard/extreme tier inside a run; this module extends
the same principle ACROSS runs: each task's prediction is voted across
N independent runs of the whole benchmark.

The vote key is `frozenset(Counter(col_sigs).items())` — identical to the
SelfConsistencyAgent voting key in `agents/self_consistency.py`. This
mirrors how the official scorer compares column multisets and ignores
column names + row order.

Tie-break: when two buckets have the same number of runs, the bucket
containing the EARLIEST run wins (deterministic with respect to run
order). When a task is missing in some runs but present in others, only
the present runs vote — a task missing from EVERY run produces no output.

CLI
---

    uv run python -m data_agent_baseline.scoring.cross_run_vote \\
        --predictions-roots run_A/output run_B/output run_C/output \\
        --output-dir voted/output
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data_agent_baseline.benchmark.schema import AnswerTable
from data_agent_baseline.scoring.normalize import column_signature

VoteKey = frozenset[tuple[frozenset[tuple[str, int]], int]]


@dataclass(frozen=True, slots=True)
class _Candidate:
    run_index: int
    prediction_path: Path
    table: AnswerTable


def _load_prediction_csv(path: Path) -> AnswerTable | None:
    if not path.is_file():
        return None
    try:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except OSError:
        return None
    if not rows:
        return None
    header = rows[0]
    body = rows[1:]
    return AnswerTable(columns=list(header), rows=[list(r) for r in body])


def _signature(table: AnswerTable) -> VoteKey:
    """Column-multiset signature ignoring column NAMES and row ORDER.

    Matches `SelfConsistencyAgent._answer_signature` exactly so the
    in-run vote and cross-run vote use the same equivalence relation.
    """
    column_sigs: list[frozenset[tuple[str, int]]] = []
    for col_idx in range(len(table.columns)):
        values = [
            str(row[col_idx]) if col_idx < len(row) else ""
            for row in table.rows
        ]
        column_sigs.append(column_signature(values))
    return frozenset(Counter(column_sigs).items())


@dataclass(frozen=True, slots=True)
class TaskVoteResult:
    task_id: str
    chosen_run_index: int | None        # None when no run had a prediction
    bucket_size: int                     # how many runs voted with the winner
    total_present_runs: int              # how many runs had any prediction
    bucket_counts: dict[int, int]        # bucket_size_value → count of buckets


def _vote_one_task(candidates: list[_Candidate]) -> int | None:
    """Return the chosen candidate's run_index, or None if no candidate."""
    if not candidates:
        return None
    buckets: dict[VoteKey, list[int]] = {}
    for cand in candidates:
        sig = _signature(cand.table)
        buckets.setdefault(sig, []).append(cand.run_index)
    # majority size, tie-break to bucket whose smallest run_index is smallest.
    winning = max(buckets.values(), key=lambda idxs: (len(idxs), -min(idxs)))
    return winning[0]


def _discover_task_ids(roots: Iterable[Path]) -> list[str]:
    """Union of `task_*` directory names across all roots."""
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("task_"):
                seen.add(child.name)
    return sorted(seen, key=lambda name: int(name.removeprefix("task_")))


def vote_across_runs(
    *,
    prediction_roots: list[Path],
    output_dir: Path,
) -> list[TaskVoteResult]:
    """Produce a single voted prediction tree under ``output_dir``.

    For each task discovered in any of the input roots:
      1. Load each run's `task_<id>/prediction.csv` (skip missing).
      2. Bucket by column-signature; pick majority (tie → earliest run).
      3. Copy that run's prediction.csv into ``output_dir/task_<id>/``.

    Returns a list of `TaskVoteResult` describing the vote outcome for
    every task — useful for diagnostics ("which tasks are stable?").
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = _discover_task_ids(prediction_roots)
    results: list[TaskVoteResult] = []

    for task_id in task_ids:
        candidates: list[_Candidate] = []
        for run_idx, root in enumerate(prediction_roots):
            pred_path = root / task_id / "prediction.csv"
            table = _load_prediction_csv(pred_path)
            if table is None:
                continue
            candidates.append(_Candidate(run_index=run_idx, prediction_path=pred_path, table=table))

        chosen = _vote_one_task(candidates)
        if chosen is None:
            results.append(
                TaskVoteResult(
                    task_id=task_id,
                    chosen_run_index=None,
                    bucket_size=0,
                    total_present_runs=0,
                    bucket_counts={},
                )
            )
            continue

        # Compute bucket statistics for diagnostics.
        buckets: dict[VoteKey, list[int]] = {}
        for cand in candidates:
            buckets.setdefault(_signature(cand.table), []).append(cand.run_index)
        winning_bucket = max(buckets.values(), key=lambda idxs: (len(idxs), -min(idxs)))
        bucket_counts: dict[int, int] = Counter(len(b) for b in buckets.values())

        # Copy the chosen prediction to the voted output tree.
        chosen_path = next(c.prediction_path for c in candidates if c.run_index == chosen)
        out_task_dir = output_dir / task_id
        out_task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chosen_path, out_task_dir / "prediction.csv")

        results.append(
            TaskVoteResult(
                task_id=task_id,
                chosen_run_index=chosen,
                bucket_size=len(winning_bucket),
                total_present_runs=len(candidates),
                bucket_counts=dict(bucket_counts),
            )
        )

    return results


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Majority-vote predictions across N benchmark runs.",
    )
    parser.add_argument(
        "--predictions-roots",
        nargs="+",
        required=True,
        type=Path,
        help="Two or more run output roots (each containing task_<id>/prediction.csv).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Where to write the voted prediction tree.",
    )
    args = parser.parse_args(argv)

    if len(args.predictions_roots) < 2:
        parser.error("--predictions-roots needs at least two roots.")

    results = vote_across_runs(
        prediction_roots=args.predictions_roots,
        output_dir=args.output_dir,
    )

    n_runs = len(args.predictions_roots)
    total = len(results)
    consensus = sum(1 for r in results if r.bucket_size == r.total_present_runs and r.bucket_size > 0)
    split = sum(1 for r in results if r.total_present_runs >= 2 and r.bucket_size < r.total_present_runs)
    no_pred = sum(1 for r in results if r.total_present_runs == 0)

    print(f"voted across {n_runs} runs → {args.output_dir}")
    print(f"  tasks total:        {total}")
    print(f"  unanimous:          {consensus}")
    print(f"  split (voted):      {split}")
    print(f"  missing every run:  {no_pred}")

    flaky = [r for r in results if r.total_present_runs >= 2 and r.bucket_size < r.total_present_runs]
    if flaky:
        print("  flaky tasks (run-disagreement):")
        for r in flaky:
            print(f"    {r.task_id:14s} bucket_size={r.bucket_size}/{r.total_present_runs} chosen=run{r.chosen_run_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
