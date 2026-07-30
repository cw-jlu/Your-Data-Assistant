#!/usr/bin/env python3
"""Phase 2 best-so-far progress charts.

The Phase 1 progress chart uses a scatter plot plus a running best envelope.
This script mirrors that view for Phase 2 full-benchmark records.

Primary data source:
  artifacts/runs/exp_14*/summary.json
  artifacts/runs/exp_14*/score_summary.json

The docs results table is used only for completed merged records that are not
represented by a full-60 artifact summary, such as the exp144 3-attempt merge.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "artifacts" / "runs"
ARTIFACT_PLOTS = REPO / "artifacts" / "plots"
DOC_PLOTS = REPO / "docs" / "plots"
PHASE2_DOC = REPO / "docs" / "PHASE_2_EXPERIMENTS.md"

RUN_PAT = re.compile(r"^exp_(?P<num>\d{3})_.+")

# These early ablations were marked as leaked in the experiment history. They
# are shown, but excluded from the clean best-so-far envelope.
LEAKED_RUNS = {
    "exp_144_answer_shape_001",
    "exp_144_prose_001",
    "exp_144_video_002",
}


@dataclass(frozen=True)
class Row:
    name: str
    mean: float
    perfect: int
    zero: int | None
    n_tasks: int
    elapsed_minutes: float | None
    source: str
    mtime: float
    is_leak: bool = False


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _numeric_exp(name: str) -> int | None:
    m = RUN_PAT.match(name)
    if not m:
        return None
    return int(m["num"])


def _row_from_run_dir(run_dir: Path) -> Row | None:
    exp_num = _numeric_exp(run_dir.name)
    if exp_num is None or exp_num < 142:
        return None

    summary = _read_json(run_dir / "summary.json") or {}
    score_summary = _read_json(run_dir / "score_summary.json") or {}

    # score_summary is authoritative when present, because exp145 stores the
    # evaluator output there and keeps only execution metadata in summary.json.
    score = score_summary if score_summary.get("mean_score") is not None else summary
    mean = score.get("mean_score")
    n_tasks = score.get("n_tasks") or summary.get("n_tasks") or summary.get("task_count")
    perfect = score.get("n_perfect") or summary.get("n_perfect")
    zero = score.get("n_zero") if score.get("n_zero") is not None else summary.get("n_zero")
    elapsed = score.get("elapsed_minutes")
    if elapsed is None:
        elapsed = summary.get("elapsed_minutes")

    if mean is None or n_tasks is None or perfect is None:
        return None
    if int(n_tasks) < 50:
        return None

    mtime_candidates = [run_dir.stat().st_mtime]
    for p in (run_dir / "summary.json", run_dir / "score_summary.json"):
        if p.is_file():
            mtime_candidates.append(p.stat().st_mtime)

    return Row(
        name=run_dir.name,
        mean=float(mean),
        perfect=int(perfect),
        zero=int(zero) if zero is not None else None,
        n_tasks=int(n_tasks),
        elapsed_minutes=float(elapsed) if elapsed is not None else None,
        source="artifact",
        mtime=max(mtime_candidates),
        is_leak=run_dir.name in LEAKED_RUNS,
    )


def _strip_markdown(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_float(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", _strip_markdown(text))
    return float(m.group(0)) if m else None


def _parse_int(text: str) -> int | None:
    m = re.search(r"\d+", _strip_markdown(text))
    return int(m.group(0)) if m else None


def _docs_table_rows(existing_names: set[str]) -> list[Row]:
    if not PHASE2_DOC.is_file():
        return []

    rows: list[Row] = []
    in_table = False
    doc_order = 0
    for line in PHASE2_DOC.read_text().splitlines():
        if line.startswith("| run |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        if line.startswith("|---"):
            continue

        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue
        run_label, _config, _att, n_text, mean_text, perfect_text, zero_text = parts[:7]
        n_tasks = _parse_int(n_text)
        mean = _parse_float(mean_text)
        perfect = _parse_int(perfect_text)
        zero = _parse_int(zero_text)
        if n_tasks is None or mean is None or perfect is None:
            continue
        if n_tasks < 50:
            continue

        clean_run_label = _strip_markdown(run_label)
        base_name = clean_run_label.split()[0]
        if base_name in existing_names:
            continue

        # Avoid duplicating artifact rows when the docs label omits only the
        # explanatory text after the run id.
        if any(name.startswith(base_name) for name in existing_names):
            if "merged" not in clean_run_label:
                continue

        doc_order += 1
        mtime = float(doc_order)
        candidates = sorted(RUNS.glob(f"{base_name}_*"))
        if candidates:
            mtime = max(p.stat().st_mtime for p in candidates)

        rows.append(
            Row(
                name=clean_run_label,
                mean=mean,
                perfect=perfect,
                zero=zero,
                n_tasks=n_tasks,
                elapsed_minutes=None,
                source="docs",
                mtime=mtime,
                is_leak=base_name in LEAKED_RUNS,
            )
        )
    return rows


def collect_rows() -> list[Row]:
    rows: list[Row] = []
    if RUNS.is_dir():
        for d in RUNS.iterdir():
            if not d.is_dir():
                continue
            row = _row_from_run_dir(d)
            if row:
                rows.append(row)

    existing_names = {r.name for r in rows}
    rows.extend(_docs_table_rows(existing_names))
    rows.sort(key=lambda r: (r.mtime, r.name))
    return rows


def _short_label(name: str) -> str:
    label = name
    label = label.replace("exp_", "e")
    replacements = [
        ("answer_shape_prose_video_keyframe_note_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard", "shape+prose+video-note+router+guard"),
        ("answer_shape_prose_pdf_preprocess_source_router_anti_agg_anti_agg_sql_guard", "shape+prose+router+guard"),
        ("answer_shape_prose_pdf_preprocess_source_router", "shape+prose+router"),
        ("answer_shape_prose_keyframes_pdf_preprocess_source_router", "shape+prose+keyframes+router"),
        ("answer_shape_prose", "shape+prose"),
        ("answer_shape", "shape"),
        ("dynamic_doc", "dynamic-doc"),
        ("phase2_baseline", "baseline"),
        ("phase2_best", "best"),
        ("pdf_preprocess", "pdf"),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    label = label.replace("_", " ")
    label = re.sub(r"\s+", " ", label).strip()
    if len(label) > 34:
        label = label[:31] + "..."
    return label


def _running_best(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    scores = values.copy()
    scores[~valid_mask] = -np.inf
    best = np.maximum.accumulate(scores)
    return np.array([np.nan if v == -np.inf else v for v in best])


def plot_score(rows: list[Row]) -> Path:
    xs = np.arange(1, len(rows) + 1)
    ys = np.array([r.mean for r in rows], dtype=float)
    leak_mask = np.array([r.is_leak for r in rows])
    docs_mask = np.array([r.source == "docs" for r in rows])
    clean_mask = ~leak_mask
    running = _running_best(ys, clean_mask)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.step(xs, running, where="post", color="#1f77b4", linewidth=2.6, alpha=0.88, label="best-so-far (clean)")
    ax.scatter(xs[clean_mask & ~docs_mask], ys[clean_mask & ~docs_mask], s=54, c="#1f77b4", alpha=0.72, edgecolors="white", linewidths=0.7, label="full benchmark")
    ax.scatter(xs[clean_mask & docs_mask], ys[clean_mask & docs_mask], s=70, c="#ff7f0e", marker="D", alpha=0.82, edgecolors="white", linewidths=0.7, label="docs-only merged record")
    if leak_mask.any():
        ax.scatter(xs[leak_mask], ys[leak_mask], s=70, c="#d62728", marker="x", linewidths=2.0, label="leak-marked ablation")

    new_best_idx: list[int] = []
    best_so_far = -math.inf
    for i, r in enumerate(rows):
        if r.is_leak:
            continue
        if r.mean > best_so_far:
            best_so_far = r.mean
            new_best_idx.append(i)
    for i in new_best_idx:
        ax.annotate(
            f"{rows[i].mean:.4f}",
            xy=(xs[i], ys[i]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#1f77b4",
            weight="bold",
        )

    ax.set_xlabel("Phase 2 full-benchmark record index", fontsize=11)
    ax.set_ylabel("lambda=0.5 mean score (60-task demo set)", fontsize=11)
    ax.set_title("kobushi Phase 2 - score per full benchmark + best-so-far", fontsize=13)
    ax.set_xticks(xs)
    ax.set_xticklabels([_short_label(r.name) for r in rows], rotation=55, ha="right", fontsize=7.5)
    y_min = max(0.25, float(np.nanmin(ys)) - 0.04)
    y_max = min(0.72, float(np.nanmax(ys)) + 0.05)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    out = ARTIFACT_PLOTS / "phase2_progress_score.png"
    ARTIFACT_PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_perfect(rows: list[Row]) -> Path:
    xs = np.arange(1, len(rows) + 1)
    ys = np.array([r.perfect for r in rows], dtype=float)
    leak_mask = np.array([r.is_leak for r in rows])
    docs_mask = np.array([r.source == "docs" for r in rows])
    clean_mask = ~leak_mask
    running = _running_best(ys, clean_mask)

    fig, ax = plt.subplots(figsize=(15, 6.0))
    ax.step(xs, running, where="post", color="#9467bd", linewidth=2.6, alpha=0.9, label="best perfect-count so far (clean)")
    ax.scatter(xs[clean_mask & ~docs_mask], ys[clean_mask & ~docs_mask], s=54, c="#9467bd", alpha=0.72, edgecolors="white", linewidths=0.7, label="full benchmark")
    ax.scatter(xs[clean_mask & docs_mask], ys[clean_mask & docs_mask], s=70, c="#ff7f0e", marker="D", alpha=0.82, edgecolors="white", linewidths=0.7, label="docs-only merged record")
    if leak_mask.any():
        ax.scatter(xs[leak_mask], ys[leak_mask], s=70, c="#d62728", marker="x", linewidths=2.0, label="leak-marked ablation")

    ax.axhline(60, color="gray", linestyle="--", alpha=0.3)
    ax.text(xs.max() + 0.3, 60, "60 tasks", fontsize=8, color="gray", va="center")

    ax.set_xlabel("Phase 2 full-benchmark record index", fontsize=11)
    ax.set_ylabel("perfect task count (out of 60)", fontsize=11)
    ax.set_title("kobushi Phase 2 - perfect-count per full benchmark + best-so-far", fontsize=13)
    ax.set_xticks(xs)
    ax.set_xticklabels([_short_label(r.name) for r in rows], rotation=55, ha="right", fontsize=7.5)
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    out = ARTIFACT_PLOTS / "phase2_progress_perfect.png"
    ARTIFACT_PLOTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def write_data(rows: list[Row]) -> Path:
    out = ARTIFACT_PLOTS / "phase2_progress_records.json"
    ARTIFACT_PLOTS.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": r.name,
            "mean_score": r.mean,
            "n_perfect": r.perfect,
            "n_zero": r.zero,
            "n_tasks": r.n_tasks,
            "elapsed_minutes": r.elapsed_minutes,
            "source": r.source,
            "is_leak": r.is_leak,
        }
        for r in rows
    ]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return out


def copy_to_docs(paths: list[Path]) -> list[Path]:
    DOC_PLOTS.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in paths:
        dst = DOC_PLOTS / path.name
        shutil.copyfile(path, dst)
        copied.append(dst)
    return copied


def main() -> int:
    rows = collect_rows()
    if not rows:
        print("no Phase 2 full-benchmark rows found")
        return 1

    score_plot = plot_score(rows)
    perfect_plot = plot_perfect(rows)
    data_path = write_data(rows)
    docs_paths = copy_to_docs([score_plot, perfect_plot])

    clean_rows = [r for r in rows if not r.is_leak]
    best = max(clean_rows, key=lambda r: r.mean)

    print(f"wrote {score_plot}")
    print(f"wrote {perfect_plot}")
    print(f"wrote {data_path}")
    for p in docs_paths:
        print(f"copied {p}")
    print(f"records plotted: {len(rows)}")
    print(f"clean records: {len(clean_rows)}")
    print(f"best clean: {best.name} = {best.mean:.4f}, perfect={best.perfect}/60")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
