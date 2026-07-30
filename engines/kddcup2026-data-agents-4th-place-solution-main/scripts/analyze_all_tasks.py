"""Aggregate per-task historical performance across all completed runs.

Walks `artifacts/runs/*/evaluation.csv` (skipping discarded / leak / partial),
computes per-task statistics, and dumps `artifacts/task_analysis/per_task_history.json`.

Output schema per task_id:
    {
        "task_id": "task_NN",
        "difficulty": "easy|medium|hard|extreme",
        "question": "...",
        "n_runs": int,
        "lambda05": {"mean": ..., "std": ..., "min": ..., "max": ...},
        "perfect_rate": float (= fraction of runs where score == 1.0),
        "zero_rate": float (= fraction where score == 0.0),
        "category": "always_solved|often_solved|variable|rarely_solved|never_solved",
        "data_files": {
            "csv": [{"name": "...", "size": N, "shape": [rows, cols]}, ...],
            "sqlite": [...],
            "json": [...],
            "doc": [...],
        },
        "gold": {"rows": int, "cols": int, "header": [...], "sample_first_3": [...]},
    }

Usage:
    uv run python scripts/analyze_all_tasks.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "artifacts" / "runs"
INPUT_ROOT = REPO / "data" / "public" / "input"
OUTPUT_ROOT = REPO / "data" / "public" / "output"
OUT_DIR = REPO / "artifacts" / "task_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "per_task_history.json"


def collect_run_scores() -> dict[str, list[float]]:
    """Collect per-task score history from all valid evaluation.csv files."""
    scores: dict[str, list[float]] = {}
    for ev in sorted(RUNS.glob("*/evaluation.csv")):
        run_name = ev.parent.name
        # Skip leak-tainted / partial / discarded runs
        if any(s in run_name for s in ["_leak", "_partial", "_502retry", "_discard"]):
            continue
        try:
            for r in csv.DictReader(open(ev)):
                tid = r.get("task_id")
                if not tid:
                    continue
                try:
                    s = float(r.get("official_score_lambda_0_5", 0) or 0)
                except (TypeError, ValueError):
                    continue
                scores.setdefault(tid, []).append(s)
        except Exception:
            continue
    return scores


def categorize(perfect_rate: float, mean_score: float) -> str:
    if perfect_rate >= 0.90:
        return "always_solved"
    if perfect_rate >= 0.50:
        return "often_solved"
    if mean_score == 0.0 and perfect_rate == 0.0:
        return "never_solved"
    if perfect_rate >= 0.10:
        return "variable"
    return "rarely_solved"


def file_summary(path: Path) -> dict:
    out: dict = {"name": path.name, "size": path.stat().st_size}
    suf = path.suffix.lower()
    try:
        if suf == ".csv":
            import pandas as pd

            df = pd.read_csv(path, nrows=1000, low_memory=False)
            # Approximate row count via size / header_bytes_per_row * nrows
            out["columns"] = list(df.columns)
            out["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
            out["sampled_shape"] = list(df.shape)
        elif suf in {".db", ".sqlite", ".sqlite3"}:
            conn = sqlite3.connect(path)
            try:
                tables = [
                    r[0]
                    for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                    ).fetchall()
                ]
                out["tables"] = []
                for t in tables:
                    cols = [
                        (r[1], r[2])
                        for r in conn.execute(f'PRAGMA table_info("{t}")').fetchall()
                    ]
                    out["tables"].append(
                        {
                            "name": t,
                            "columns": [c[0] for c in cols],
                            "dtypes": {c[0]: c[1] for c in cols},
                        }
                    )
            finally:
                conn.close()
        elif suf == ".json":
            with path.open("rb") as fh:
                head = fh.read(8192).decode("utf-8", errors="replace")
            out["head_preview"] = head[:300]
        elif suf in {".md", ".txt"}:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                out["head_preview"] = fh.read(500)
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def collect_task_files(task_dir: Path) -> dict:
    out = {"csv": [], "sqlite": [], "json": [], "doc": []}
    context = task_dir / "context"
    if not context.is_dir():
        return out
    for p in sorted(context.rglob("*")):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        info = file_summary(p)
        info["rel_path"] = str(p.relative_to(context))
        if suf == ".csv":
            out["csv"].append(info)
        elif suf in {".db", ".sqlite", ".sqlite3"}:
            out["sqlite"].append(info)
        elif suf == ".json":
            out["json"].append(info)
        elif suf in {".md", ".txt"}:
            out["doc"].append(info)
    return out


def collect_gold(task_id: str) -> dict:
    g = OUTPUT_ROOT / task_id / "gold.csv"
    if not g.is_file():
        return {}
    rows = list(csv.reader(open(g)))
    if not rows:
        return {"rows": 0, "cols": 0, "header": [], "sample_first_3": []}
    return {
        "rows": len(rows) - 1,
        "cols": len(rows[0]),
        "header": rows[0],
        "sample_first_3": rows[1:4],
    }


def collect_task_meta(task_id: str) -> dict:
    p = INPUT_ROOT / task_id / "task.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def main():
    scores = collect_run_scores()
    print(f"collected scores for {len(scores)} tasks")

    out: list[dict] = []
    for task_dir in sorted(INPUT_ROOT.iterdir(), key=lambda p: int(p.name.removeprefix("task_"))):
        tid = task_dir.name
        s = scores.get(tid, [])
        meta = collect_task_meta(tid)
        gold = collect_gold(tid)
        files = collect_task_files(task_dir)

        if s:
            mean = statistics.fmean(s)
            std = statistics.stdev(s) if len(s) >= 2 else 0.0
            perfect_rate = sum(1 for x in s if x == 1.0) / len(s)
            zero_rate = sum(1 for x in s if x == 0.0) / len(s)
        else:
            mean = std = perfect_rate = zero_rate = 0.0
        cat = categorize(perfect_rate, mean) if s else "no_data"
        out.append(
            {
                "task_id": tid,
                "difficulty": meta.get("difficulty", "?"),
                "question": meta.get("question", ""),
                "n_runs": len(s),
                "lambda05": {
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "min": round(min(s), 4) if s else None,
                    "max": round(max(s), 4) if s else None,
                },
                "perfect_rate": round(perfect_rate, 3),
                "zero_rate": round(zero_rate, 3),
                "category": cat,
                "gold": gold,
                "data_files": files,
            }
        )

    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    # Print category counts
    from collections import Counter

    cats = Counter(t["category"] for t in out)
    print(f"wrote {OUT_JSON}")
    for c in ["always_solved", "often_solved", "variable", "rarely_solved", "never_solved", "no_data"]:
        if cats.get(c):
            print(f"  {c}: {cats[c]} tasks")


if __name__ == "__main__":
    main()
