"""Generate the skeleton + per-task sections of `docs/PUBLIC_TASK_ANALYSIS.md`.

Reads `artifacts/task_analysis/per_task_history.json`. Per-task narrative
("required reasoning steps", "failure modes", "memo") is left as TODO blocks
that a follow-up pass (manually or with another agent run) will fill in.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "artifacts" / "task_analysis" / "per_task_history.json"
OUT = REPO / "docs" / "PUBLIC_TASK_ANALYSIS.md"


_CAT_LABEL = {
    "always_solved": "🟢 常に解ける (always-solved, ≥90% perfect)",
    "often_solved": "🟡 ほぼ解ける (often-solved, 50-90% perfect)",
    "variable": "🟠 揺れる (variable, 10-50% perfect)",
    "rarely_solved": "🔴 滅多に解けない (rarely-solved, 1-10% perfect)",
    "never_solved": "⚫ 一度も解けない (never-solved, 0% perfect)",
}
_CAT_ORDER = ["never_solved", "rarely_solved", "variable", "often_solved", "always_solved"]


def _files_summary_md(files: dict) -> str:
    lines = []
    for kind, items in files.items():
        if not items:
            continue
        for item in items:
            size_kb = item.get("size", 0) / 1024
            base = f"`{item.get('rel_path','?')}` ({size_kb:.1f}KB)"
            if kind == "csv":
                cols = item.get("columns") or []
                shape = item.get("sampled_shape") or [None, None]
                line = f"- **CSV** {base}: shape sampled={shape}, cols=[{', '.join(cols[:8])}{'...' if len(cols) > 8 else ''}]"
            elif kind == "sqlite":
                tables = item.get("tables") or []
                tnames = [t["name"] for t in tables]
                line = f"- **SQLite** {base}: tables={tnames}"
                lines.append(line)
                for t in tables:
                    cols = t.get("columns", [])
                    line = f"    - table `{t['name']}`: cols=[{', '.join(cols[:10])}{'...' if len(cols) > 10 else ''}]"
                    lines.append(line)
                continue
            elif kind == "json":
                line = f"- **JSON** {base}"
            else:
                line = f"- **DOC** {base}"
            lines.append(line)
    return "\n".join(lines) if lines else "- (no data files)"


def _gold_md(gold: dict) -> str:
    if not gold:
        return "- (no gold data)"
    rows = gold.get("rows", 0)
    cols = gold.get("cols", 0)
    header = gold.get("header", [])
    samples = gold.get("sample_first_3", [])
    lines = [f"- shape: **{rows} rows × {cols} cols**", f"- header: `{header}`"]
    if samples:
        lines.append("- first rows:")
        for row in samples:
            lines.append(f"  - `{row}`")
    return "\n".join(lines)


def _stats_line(t: dict) -> str:
    s = t["lambda05"]
    return (
        f"n={t['n_runs']}, "
        f"λ0.5 mean={s['mean']}, "
        f"std={s['std']}, "
        f"perfect_rate={t['perfect_rate']:.0%}, "
        f"zero_rate={t['zero_rate']:.0%}"
    )


def _section_for_task(t: dict, depth_marker: str) -> str:
    body = []
    body.append(f"### {t['task_id']}  ({t['difficulty']}, {depth_marker})")
    body.append("")
    body.append(f"**履歴統計**: {_stats_line(t)}")
    body.append("")
    body.append(f"**質問**:")
    body.append("> " + (t["question"] or "(empty)"))
    body.append("")
    body.append("**入力データ**:")
    body.append(_files_summary_md(t["data_files"]))
    body.append("")
    body.append("**Gold answer**:")
    body.append(_gold_md(t["gold"]))
    body.append("")
    body.append("**正解までの reasoning steps**: <!-- TODO: fill per-task -->")
    body.append("")
    body.append("**失敗モード / 過去 trace の傾向**: <!-- TODO: fill per-task -->")
    body.append("")
    body.append("**メモ**: <!-- TODO: fill per-task -->")
    body.append("")
    return "\n".join(body)


def main():
    data = json.loads(SOURCE.read_text())
    by_cat: dict[str, list[dict]] = {c: [] for c in _CAT_ORDER}
    for t in data:
        by_cat.setdefault(t["category"], []).append(t)
    # sort within category by perfect_rate ascending (= harder first)
    for c, ts in by_cat.items():
        ts.sort(key=lambda x: (x["perfect_rate"], x["lambda05"]["mean"]))

    md = []
    md.append("# DABench Public 50 タスク全分析")
    md.append("")
    md.append("> 生成: `scripts/build_task_analysis_md.py` (基礎データ: `scripts/analyze_all_tasks.py`)")
    md.append("> 履歴ソース: `artifacts/runs/*/evaluation.csv` (leak / partial / 502retry を除外)")
    md.append("")
    # Summary table
    md.append("## カテゴリ分布")
    md.append("")
    md.append("| カテゴリ | 説明 | task 数 |")
    md.append("|---|---|---:|")
    for c in _CAT_ORDER:
        md.append(f"| {c} | {_CAT_LABEL[c]} | {len(by_cat.get(c, []))} |")
    md.append("")
    md.append("## TOC")
    md.append("")
    for c in _CAT_ORDER:
        if not by_cat.get(c):
            continue
        md.append(f"- **{_CAT_LABEL[c]}** ({len(by_cat[c])} task)")
        for t in by_cat[c]:
            md.append(f"  - [{t['task_id']}](#{t['task_id'].replace('_','-')})  {t['difficulty']} · n={t['n_runs']} · mean={t['lambda05']['mean']} · perfect={t['perfect_rate']:.0%}")
    md.append("")
    md.append("---")
    md.append("")
    # Per-category sections
    for c in _CAT_ORDER:
        items = by_cat.get(c, [])
        if not items:
            continue
        md.append(f"## {_CAT_LABEL[c]}  ({len(items)} task)")
        md.append("")
        for t in items:
            depth = (
                "深掘り" if c in ("never_solved", "rarely_solved") else
                "中" if c == "variable" else
                "簡潔"
            )
            md.append(_section_for_task(t, depth))
            md.append("---")
            md.append("")
    # Cross-cutting / placeholders
    md.append("## 横断的失敗パターン (Phase 3)")
    md.append("")
    md.append("<!-- TODO: aggregate failure modes from per-task analysis -->")
    md.append("")
    md.append("## 推奨改善軸 (Phase 3)")
    md.append("")
    md.append("<!-- TODO: derive from cross-cutting patterns -->")
    md.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md))
    print(f"wrote {OUT}  ({len(md)} lines)")


if __name__ == "__main__":
    main()
