#!/usr/bin/env python3
"""Generate site/data.json + copy plot PNGs for the static dashboard.

Output goes to site/ (served by GitHub Pages from main / site).

Run after each takt commit. Reads:
- artifacts/replications/<exp>/summary_*.json (latest per exp)
- artifacts/runs/<exp>_NNN/evaluation.json (single-run scores)
- LEADERBOARD.md (parses the submission table)
- artifacts/plots/progress_*.png

Writes:
- site/data.json
- site/plots/progress_score.png
- site/plots/progress_perfect.png
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "artifacts" / "runs"
REPL = REPO / "artifacts" / "replications"
PLOTS_SRC = REPO / "artifacts" / "plots"
DOCS = REPO / "site"
DOCS_PLOTS = DOCS / "plots"

DOCS.mkdir(exist_ok=True)
DOCS_PLOTS.mkdir(exist_ok=True)

RUN_PAT = re.compile(r"^(?P<name>exp_(?P<num>\d{3})_.+)_(?P<idx>\d{3})$")


def first_eval(run_dirs):
    for d in sorted(run_dirs):
        ej = d / "evaluation.json"
        if not ej.is_file():
            continue
        try:
            data = json.loads(ej.read_text())
        except Exception:
            continue
        return {
            "run_dir": d.name,
            "lambda_0_5": data.get("official_score_lambda_0_5_mean", 0.0),
            "perfect": data.get("perfect_recall_no_extras_count", 0),
            "missing": data.get("missing_prediction_count", 0),
            "zero_recall": data.get("zero_recall_count", 0),
        }
    return None


def latest_summary(exp_dir):
    summaries = sorted(exp_dir.glob("summary_*.json"))
    if not summaries:
        return None
    try:
        return json.loads(summaries[-1].read_text())
    except Exception:
        return None


def collect_experiments():
    """Group runs by exp name; pick first eval + latest replication summary."""
    by_name = {}
    for d in sorted(RUNS.iterdir() if RUNS.is_dir() else []):
        m = RUN_PAT.match(d.name)
        if not m:
            continue
        name = m.group("name")
        by_name.setdefault(name, []).append(d)

    out = []
    for name, dirs in by_name.items():
        m = RUN_PAT.match(dirs[0].name)
        num = int(m.group("num"))
        single = first_eval(dirs)
        repl_dir = REPL / name
        repl = latest_summary(repl_dir) if repl_dir.is_dir() else None

        # Detect attempts class from runner.py _ATTEMPT_TEMPS literal
        attempts_class = None
        runner_py = REPO / "src" / "experiments" / name / "runner.py"
        if runner_py.is_file():
            try:
                import re as _re
                m_at = _re.search(r"_ATTEMPT_TEMPS[^=]*=\s*\(([^)]*)\)", runner_py.read_text())
                if m_at:
                    temps = [t.strip() for t in m_at.group(1).split(",") if t.strip()]
                    attempts_class = len(temps)
            except Exception:
                pass

        entry = {
            "exp_num": num,
            "name": name,
            "n_runs": len(dirs),
            "attempts_class": attempts_class,  # 1 = single-attempt, 3 = union, etc.
            "single_run_score": single["lambda_0_5"] if single else None,
            "single_run_perfect": single["perfect"] if single else None,
            "single_run_missing": single["missing"] if single else None,
        }
        if repl:
            entry["repl_n"] = repl["n"]
            entry["repl_mean"] = repl["lambda_0_5"]["mean"]
            entry["repl_std"] = repl["lambda_0_5"]["std"]
            entry["repl_median"] = repl["lambda_0_5"]["median"]
            entry["repl_ci_lo"] = repl["lambda_0_5"]["ci_95"][0]
            entry["repl_ci_hi"] = repl["lambda_0_5"]["ci_95"][1]
            entry["repl_missing_mean"] = repl["missing"]["mean"]
        else:
            entry["repl_n"] = None
        out.append(entry)
    out.sort(key=lambda e: e["exp_num"])
    return out


def parse_leaderboard():
    """Parse the markdown table in LEADERBOARD.md."""
    md = REPO / "LEADERBOARD.md"
    if not md.is_file():
        return []
    text = md.read_text()
    rows = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| ver "):
            in_table = True
            continue
        if in_table and line.startswith("|----"):
            continue
        if in_table:
            if not line.startswith("|"):
                in_table = False
                continue
            cells = [c.strip().strip("*") for c in line.split("|")[1:-1]]
            if len(cells) >= 6:
                rows.append({
                    "version": cells[0].strip("*").strip(),
                    "date": cells[1],
                    "exp": cells[2].strip("`").strip(),
                    "local_1run": cells[3],
                    "local_n_mean": cells[4],
                    "local_n": cells[5] if len(cells) > 5 else "",
                    "lb": cells[6] if len(cells) > 6 else "",
                    "rank": cells[7] if len(cells) > 7 else "",
                    "gap": cells[8] if len(cells) > 8 else "",
                })
    return rows


def get_git_info():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO).decode().strip()
        msg = subprocess.check_output(["git", "log", "-1", "--pretty=%s"], cwd=REPO).decode().strip()
        ts = subprocess.check_output(["git", "log", "-1", "--pretty=%ct"], cwd=REPO).decode().strip()
        return {"sha": sha, "msg": msg, "commit_ts": int(ts)}
    except Exception:
        return {"sha": "?", "msg": "?", "commit_ts": 0}


def main() -> None:
    experiments = collect_experiments()
    submissions = parse_leaderboard()

    # Best-so-far metrics — split by attempts class (single-attempt vs union)
    # because single-attempt has ~+0.04 lower floor than 3-attempt union
    # (memo: feedback_attempts_class.md).
    best_repl = None
    best_repl_single = None  # best replication with attempts_class == 1
    best_repl_union = None   # best replication with attempts_class >= 2
    for e in experiments:
        if e.get("repl_mean") is None:
            continue
        if best_repl is None or e["repl_mean"] > best_repl["repl_mean"]:
            best_repl = e
        ac = e.get("attempts_class")
        if ac == 1:
            if best_repl_single is None or e["repl_mean"] > best_repl_single["repl_mean"]:
                best_repl_single = e
        elif ac and ac >= 2:
            if best_repl_union is None or e["repl_mean"] > best_repl_union["repl_mean"]:
                best_repl_union = e
    best_single = max(
        (e for e in experiments if e.get("single_run_score") is not None),
        key=lambda e: e["single_run_score"],
        default=None,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "stats": {
            "n_experiments": len(experiments),
            "n_with_replication": sum(1 for e in experiments if e.get("repl_n")),
            "best_single_run_score": best_single["single_run_score"] if best_single else None,
            "best_single_run_exp": best_single["name"] if best_single else None,
            "best_repl_mean": best_repl["repl_mean"] if best_repl else None,
            "best_repl_exp": best_repl["name"] if best_repl else None,
            "best_repl_std": best_repl["repl_std"] if best_repl else None,
            # Per-attempts-class bests (= the meaningful comparison):
            "best_repl_single_mean": best_repl_single["repl_mean"] if best_repl_single else None,
            "best_repl_single_exp": best_repl_single["name"] if best_repl_single else None,
            "best_repl_single_std": best_repl_single["repl_std"] if best_repl_single else None,
            "best_repl_union_mean": best_repl_union["repl_mean"] if best_repl_union else None,
            "best_repl_union_exp": best_repl_union["name"] if best_repl_union else None,
            "best_repl_union_std": best_repl_union["repl_std"] if best_repl_union else None,
        },
        "submissions": submissions,
        "experiments": experiments,
    }

    (DOCS / "data.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Copy plots
    for f in ("progress_score.png", "progress_perfect.png"):
        src = PLOTS_SRC / f
        if src.is_file():
            shutil.copy2(src, DOCS_PLOTS / f)

    print(f"site/data.json: {len(experiments)} experiments, {len(submissions)} submissions")
    print(f"  best repl (overall):  {best_repl['name'] if best_repl else '—'} (n={best_repl['repl_n'] if best_repl else 0}) mean={best_repl['repl_mean'] if best_repl else 0:.4f}")
    if best_repl_single:
        print(f"  best repl (single):   {best_repl_single['name']} (n={best_repl_single['repl_n']}) mean={best_repl_single['repl_mean']:.4f}")
    if best_repl_union:
        print(f"  best repl (union):    {best_repl_union['name']} (n={best_repl_union['repl_n']}) mean={best_repl_union['repl_mean']:.4f}")


if __name__ == "__main__":
    main()
