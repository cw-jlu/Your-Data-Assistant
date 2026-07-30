#!/usr/bin/env python3
"""AutoML-style best-so-far progress chart.

X axis: experiment number (chronological iteration).
Y axis: λ0.5 score (or perfect count).
Each experiment plotted as a point. Running maximum drawn as a step
function — the iconic "stair-step" curve showing how the best-known
score climbed over iterations.

Outputs to artifacts/plots/progress_<metric>.png
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "artifacts" / "runs"
REPL = REPO / "artifacts" / "replications"
PLOTS = REPO / "artifacts" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

RUN_PAT = re.compile(r"^(?P<name>exp_(?P<num>\d{3})_.+)_(?P<idx>\d{3})$")

# Leak-contaminated exps excluded from running max (still plotted as crosses).
LEAK = {
    "exp_017_tie_aware_superlative",
    "exp_023_glossary_preamble",
    "exp_024_glossary_v2",
    "exp_026_fewshot_pathfix",
    "exp_027_loop_break",
    "exp_029_schema_first",
}


def first_eval(run_dirs: list[Path]) -> dict | None:
    """Return the FIRST (oldest) full evaluation — preserves the historical
    1-run "what we saw at decision time" rather than later replication runs.

    Falls back to summary.json (= newer run schema) when evaluation.json
    is absent.
    """
    for d in sorted(run_dirs):
        ej = d / "evaluation.json"
        if ej.is_file():
            try:
                ev = json.loads(ej.read_text())
            except Exception:
                ev = None
            if ev and ev.get("task_count", 0) >= 50:
                return ev
        sj = d / "summary.json"
        if sj.is_file():
            try:
                s = json.loads(sj.read_text())
            except Exception:
                s = None
            if s and s.get("n_tasks", 0) >= 50:
                return {
                    "task_count": s["n_tasks"],
                    "official_score_lambda_0_5_mean": float(s.get("mean_score", 0.0)),
                    "perfect_recall_no_extras_count": int(s.get("n_perfect", 0)),
                }
    return None


def latest_replication_mean(name: str) -> tuple[float, float, int] | None:
    rep_dir = REPL / name
    if rep_dir.is_dir():
        summaries = sorted(rep_dir.glob("summary_*.json"))
        if summaries:
            try:
                s = json.loads(summaries[-1].read_text())
                l = s.get("lambda_0_5", {})
                return float(l.get("mean")), float(l.get("std", 0.0)), int(s.get("n", 0))
            except Exception:
                pass
    # Fallback: aggregate ≥3 summary.json runs from artifacts/runs/<name>_NNN.
    means: list[float] = []
    for d in sorted(RUNS.glob(f"{name}_*")):
        sj = d / "summary.json"
        if not sj.is_file():
            continue
        try:
            s = json.loads(sj.read_text())
            if s.get("n_tasks", 0) >= 50 and "mean_score" in s:
                means.append(float(s["mean_score"]))
        except Exception:
            continue
    if len(means) < 3:
        return None
    mean = sum(means) / len(means)
    var = sum((m - mean) ** 2 for m in means) / max(1, len(means) - 1)
    return mean, var ** 0.5, len(means)


_ATTEMPT_TEMPS_RE = re.compile(r"_ATTEMPT_TEMPS[^=]*=\s*\(([^)]*)\)")


def detect_attempts_class(name: str) -> int | None:
    """Parse src/experiments/<name>/runner.py for the _ATTEMPT_TEMPS literal.

    Returns 1 for single-attempt, n for union/multi-attempt. None if unknown.
    """
    runner_py = REPO / "src" / "experiments" / name / "runner.py"
    if not runner_py.is_file():
        return None
    try:
        m = _ATTEMPT_TEMPS_RE.search(runner_py.read_text())
        if not m:
            return None
        temps = [t.strip() for t in m.group(1).split(",") if t.strip()]
        return len(temps) or None
    except Exception:
        return None


def collect_data() -> list[dict]:
    by_exp: dict[str, list[Path]] = {}
    for d in sorted(RUNS.iterdir() if RUNS.is_dir() else []):
        if not d.is_dir():
            continue
        m = RUN_PAT.match(d.name)
        if not m:
            continue
        by_exp.setdefault(m["name"], []).append(d)

    rows: list[dict] = []
    for name in sorted(by_exp.keys()):
        m = RUN_PAT.match(by_exp[name][0].name)
        num = int(m["num"])
        ev = first_eval(by_exp[name])
        if not ev:
            continue
        single = float(ev["official_score_lambda_0_5_mean"])
        perfect = int(ev["perfect_recall_no_extras_count"])
        rep = latest_replication_mean(name)
        attempts = detect_attempts_class(name)
        rows.append(
            {
                "num": num,
                "name": name,
                "single": single,
                "perfect": perfect,
                "rep_mean": rep[0] if rep else None,
                "rep_std": rep[1] if rep else None,
                "rep_n": rep[2] if rep else None,
                "is_leak": name in LEAK,
                "attempts_class": attempts,
            }
        )
    rows.sort(key=lambda r: r["num"])
    return rows


# LB linear fit from our 4 submissions (v1, v2, v5, v11)
# local mean = (0.6882, 0.7353, 0.7950, 0.8130)
# LB       = (0.4526, 0.4969, 0.5789, 0.5965)
# LB ≈ 1.188 * local − 0.369 (= least-squares); gap ≈ −0.217 ± 0.011 (stable)
LB_SLOPE = 1.188
LB_INTERCEPT = -0.369

# Actual LB submissions: (exp_num, local_mean, LB)
LB_SUBMISSIONS = [
    (68, 0.6882, 0.4526, "v1"),
    (86, 0.7353, 0.4969, "v2"),
    (122, 0.7950, 0.5789, "v5"),
    (137, 0.8130, 0.5965, "v11"),
]


def plot_score(rows: list[dict]) -> Path:
    xs = np.array([r["num"] for r in rows])
    ys = np.array([r["single"] for r in rows])
    leak_mask = np.array([r["is_leak"] for r in rows])
    rep_mask = np.array([r["rep_mean"] is not None for r in rows])
    rep_means = np.array([r["rep_mean"] if r["rep_mean"] else np.nan for r in rows])
    rep_stds = np.array([r["rep_std"] if r["rep_std"] else np.nan for r in rows])
    attempts_arr = np.array([r["attempts_class"] or -1 for r in rows])

    # "Effective" score per row: prefer rep_mean if available (= more reliable
    # than 1-run), else fall back to single-run score. Leaks excluded.
    eff = np.where(rep_mask, rep_means, ys)
    eff[leak_mask] = -np.inf

    def running_max_for(mask: np.ndarray) -> np.ndarray:
        scores = eff.copy()
        scores[~mask] = -np.inf
        rm = np.maximum.accumulate(scores)
        last = -np.inf
        cleaned = []
        for v in rm:
            if v == -np.inf:
                cleaned.append(np.nan)
            else:
                last = v
                cleaned.append(v)
        return np.array(cleaned)

    # Single best-so-far envelope across all experiments (rep_mean preferred,
    # fall back to single-run; leaks excluded).
    rm_overall = running_max_for(np.ones_like(eff, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.step(xs, rm_overall, where="post", color="#1f77b4",
            linewidth=2.5, label="best-so-far (rep_mean preferred, leak-excluded)", alpha=0.85)

    # All single-run scatter
    clean_idx = ~leak_mask & ~rep_mask
    leak_idx = leak_mask
    rep_idx = rep_mask & ~leak_mask

    ax.scatter(xs[clean_idx], ys[clean_idx], s=38, c="#1f77b4",
               alpha=0.55, edgecolors="white", linewidths=0.5,
               label="single-run (clean)", zorder=3)
    if leak_idx.any():
        ax.scatter(xs[leak_idx], ys[leak_idx], s=46, c="#d62728",
                   marker="x", linewidths=2.0, label="leak-contaminated (excluded)",
                   zorder=4)

    # Replication points: error bars + bigger marker
    if rep_idx.any():
        ax.errorbar(xs[rep_idx], rep_means[rep_idx], yerr=rep_stds[rep_idx],
                    fmt="o", markersize=10, color="#2ca02c",
                    ecolor="#2ca02c", elinewidth=1.5, capsize=4,
                    markerfacecolor="white", markeredgewidth=2,
                    label="n=3 mean ± std (replicated)", zorder=5)
        # Annotate replicated exps
        for i in np.where(rep_idx)[0]:
            ax.annotate(
                f" {rows[i]['name'].split('_', 2)[-1][:12]}",
                xy=(xs[i], rep_means[i]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=8, color="#2ca02c",
            )

    # Reference line: current verified record (= latest submitted version's local n-mean)
    ax.axhline(0.8130, color="#2ca02c", linestyle=":", alpha=0.6, linewidth=1.2)
    ax.text(xs.max() + 0.5, 0.8130, "0.8130  exp_137 n=5 mean (v11, LB=0.5965)",
            fontsize=8, color="#2ca02c", va="center", weight="bold")

    ax.set_xlabel("Experiment number (chronological)", fontsize=11)
    ax.set_ylabel("λ0.5 score (public 50, local)", fontsize=11)
    ax.set_title(
        "kobushi — λ0.5 score per experiment + best-so-far envelope\n"
        f"(N={len(rows)} experiments, replicated highlighted, LB submissions overlaid)",
        fontsize=12,
    )
    ax.set_ylim(0.3, 0.85)
    ax.set_xlim(xs.min() - 1, xs.max() + 8)
    ax.grid(True, alpha=0.3)

    # Right axis: actual LB score (= same scale 0.3-0.85), LB submissions plotted at real LB y-position
    ax_lb = ax.twinx()
    ax_lb.set_ylim(0.3, 0.85)
    lb_x = [s[0] for s in LB_SUBMISSIONS]
    lb_y = [s[2] for s in LB_SUBMISSIONS]  # actual LB scores
    ax_lb.plot(lb_x, lb_y, color="#d62728", linestyle="-", linewidth=2.2,
               marker="D", markersize=10, markerfacecolor="#d62728",
               markeredgecolor="black", markeredgewidth=1.0,
               label="actual LB submissions (v1/v2/v5/v11)", zorder=8)
    for exp_num, _local, lb, label in LB_SUBMISSIONS:
        ax_lb.annotate(f"  {label}: {lb:.4f}", xy=(exp_num, lb),
                       xytext=(8, -12), textcoords="offset points",
                       fontsize=8.5, color="#d62728", weight="bold")
    ax_lb.set_ylabel("LB score", fontsize=10, color="#d62728")
    ax_lb.tick_params(axis="y", colors="#d62728")

    # Combine legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_lb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9, framealpha=0.95)

    out = PLOTS / "progress_score.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_perfect(rows: list[dict]) -> Path:
    xs = np.array([r["num"] for r in rows])
    ys = np.array([r["perfect"] for r in rows])
    leak_mask = np.array([r["is_leak"] for r in rows])

    eff = ys.copy().astype(float)
    eff[leak_mask] = -np.inf
    running_max = np.maximum.accumulate(eff)
    cleaned = []
    last = -np.inf
    for v in running_max:
        if v == -np.inf:
            cleaned.append(np.nan)
        else:
            last = v
            cleaned.append(v)
    running_max_arr = np.array(cleaned)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.step(xs, running_max_arr, where="post", color="#9467bd",
            linewidth=2.5, label="best perfect-count so far (leak-excluded)")

    clean_idx = ~leak_mask
    ax.scatter(xs[clean_idx], ys[clean_idx], s=38, c="#9467bd",
               alpha=0.55, edgecolors="white", linewidths=0.5,
               label="perfect-count (clean)", zorder=3)
    if leak_mask.any():
        ax.scatter(xs[leak_mask], ys[leak_mask], s=46, c="#d62728",
                   marker="x", linewidths=2.0, label="leak-contaminated",
                   zorder=4)

    ax.axhline(50, color="gray", linestyle="--", alpha=0.3)
    ax.text(xs.max() + 0.3, 50, "50 (theoretical max)", fontsize=8, color="gray", va="center")
    ax.axhline(38, color="#1f77b4", linestyle=":", alpha=0.4)
    ax.text(xs.max() + 0.3, 38,
            "38 (= 50 - 12 always-fail)", fontsize=8, color="#1f77b4", va="center")

    ax.set_xlabel("Experiment number", fontsize=11)
    ax.set_ylabel("perfect_recall_no_extras_count (out of 50)", fontsize=11)
    ax.set_title("kobushi — perfect-count per experiment + best-so-far", fontsize=12)
    ax.set_ylim(0, 50)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    out = PLOTS / "progress_perfect.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    rows = collect_data()
    if not rows:
        print("no data")
        return 1
    p1 = plot_score(rows)
    p2 = plot_perfect(rows)
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print(f"total experiments plotted: {len(rows)}")
    # Brief summary
    clean = [r for r in rows if not r["is_leak"]]
    print(f"clean (leak-free): {len(clean)} exps")
    best = max(clean, key=lambda r: r["single"])
    print(f"best 1-run: {best['name']} = {best['single']:.4f}")
    rep_rows = [r for r in clean if r["rep_mean"] is not None]
    if rep_rows:
        best_rep = max(rep_rows, key=lambda r: r["rep_mean"])
        print(f"best n=3 mean: {best_rep['name']} = {best_rep['rep_mean']:.4f} ± {best_rep['rep_std']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
