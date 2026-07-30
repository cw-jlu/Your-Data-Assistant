#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
DEFAULT_BULL_ROOTS = [
    Path("/tmp/phase2_source_check/BULL_full/BULL/BULL-cn"),
    Path("/tmp/phase2_source_check/BULL_full/BULL/BULL-en"),
]
PHASE2_INPUT = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "input"
PHASE2_OUTPUT = REPO / "data" / "phase2_demo" / "demo_samples_phase2" / "output"


CUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("record_data", re.compile(r"记录|数据|是什么样|看一下|显示|展示|列出|查一下|搜一下")),
    ("entity_list", re.compile(r"哪些|有哪些|哪几|哪个|名称|简称|公司|基金|经理|股东")),
    ("count", re.compile(r"数目|数量|多少|几只|几家|多少个|共有|总共有")),
    ("distribution", re.compile(r"分布|按.+分组|分组展示")),
    ("average", re.compile(r"平均|均值|average", re.I)),
    ("rank_or_extreme", re.compile(r"最高|最大|最低|最小|排名|前\\d+|top", re.I)),
    ("return_rate", re.compile(r"回报率|收益率|回报|收益|return rate", re.I)),
    ("code", re.compile(r"代码|证券代码|股票代码|基金代码|code", re.I)),
    ("date_or_period", re.compile(r"日期|时间|年度|年份|期间|周期|date|year|period", re.I)),
]


DATE_COLUMNS = {"enddate", "tradingday", "listeddate", "inceptiondate", "establishmentdate"}
ID_COLUMNS = {
    "id",
    "innercode",
    "companycode",
    "personalcode",
    "secucode",
    "astockcode",
    "fundcode",
}
DISPLAY_COLUMNS = {
    "chinameabbr",
    "ashareabbr",
    "secuabbr",
    "abbrchiname",
    "chinesename",
    "name",
    "fund",
    "shname",
    "fpshname",
    "investadvisorabbrname",
}


@dataclass(frozen=True)
class SelectItem:
    table: str
    column: str
    agg: str
    distinct: bool

    @property
    def rendered(self) -> str:
        base = self.column if not self.table else f"{self.table}.{self.column}"
        if self.agg:
            base = f"{self.agg.upper()}({base})"
        if self.distinct:
            base = f"DISTINCT {base}"
        return base

    @property
    def column_lower(self) -> str:
        return self.column.lower()


def _task_sort_key(task_id: str) -> int:
    try:
        return int(task_id.split("_", 1)[1])
    except Exception:
        return 10**9


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_bull_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        language = root.name.replace("BULL-", "")
        for name in ("train.json", "dev.json"):
            path = root / name
            if path.is_file():
                for row in _load_json(path):
                    row = dict(row)
                    row["_bull_language"] = language
                    row["_bull_file"] = str(path)
                    rows.append(row)
    return rows


def _select_items(row: dict[str, Any]) -> list[SelectItem]:
    select = row.get("select")
    if not isinstance(select, list):
        return []
    items: list[SelectItem] = []
    for raw in select[1:]:
        if not isinstance(raw, list) or len(raw) < 4:
            continue
        agg, table, column, distinct = raw[:4]
        items.append(
            SelectItem(
                table=str(table or ""),
                column=str(column or ""),
                agg=str(agg or ""),
                distinct=bool(distinct),
            )
        )
    return items


def _cues(question: str) -> list[str]:
    hits = [name for name, pattern in CUE_PATTERNS if pattern.search(question)]
    return hits or ["uncategorized"]


def _question_kind(items: list[SelectItem]) -> str:
    if any(item.agg.upper() == "COUNT" for item in items):
        return "count_or_distribution"
    if any(item.agg for item in items):
        return "aggregate_metric"
    if len(items) == 1:
        return "single_output"
    return "multi_output"


def _has_date_output(items: list[SelectItem]) -> bool:
    return any(item.column_lower in DATE_COLUMNS or "date" in item.column_lower for item in items)


def _has_id_output(items: list[SelectItem]) -> bool:
    return any(item.column_lower in ID_COLUMNS or item.column_lower.endswith("code") for item in items)


def _has_display_output(items: list[SelectItem]) -> bool:
    return any(item.column_lower in DISPLAY_COLUMNS or "abbr" in item.column_lower for item in items)


def _read_gold_header(task_id: str) -> list[str]:
    path = PHASE2_OUTPUT / task_id / "gold.csv"
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0] if rows else []


def _load_phase2_questions() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for path in sorted(PHASE2_INPUT.glob("task_*/task.json"), key=lambda p: _task_sort_key(p.parent.name)):
        payload = _load_json(path)
        task_id = str(payload["task_id"])
        out.append({"task_id": task_id, "question": str(payload["question"])})
    return out


def _char_ngrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    grams = {compact[i : i + 2] for i in range(max(len(compact) - 1, 0))}
    grams.update(re.findall(r"[a-z0-9_]+", text.lower()))
    return {gram for gram in grams if gram}


def _best_bull_match(question: str, rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    question_grams = _char_ngrams(question)
    shortlist: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        row_grams = row.get("_question_grams")
        if not isinstance(row_grams, set):
            row_grams = _char_ngrams(str(row["question"]))
            row["_question_grams"] = row_grams
        union = len(question_grams | row_grams)
        rough = len(question_grams & row_grams) / union if union else 0.0
        if rough:
            shortlist.append((rough, row))
    shortlist.sort(key=lambda item: item[0], reverse=True)

    best_score = 0.0
    best_row: dict[str, Any] | None = None
    for _, row in shortlist[:50]:
        score = difflib.SequenceMatcher(None, question, str(row["question"])).ratio()
        if score > best_score:
            best_score = score
            best_row = row
    return best_score, best_row


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pct(num: int, den: int) -> str:
    if not den:
        return "0.0%"
    return f"{100 * num / den:.1f}%"


def _make_markdown(
    *,
    rows: list[dict[str, Any]],
    cue_rows: list[dict[str, Any]],
    phase2_rows: list[dict[str, Any]],
    top_column_rows: list[dict[str, Any]],
) -> str:
    total = len(rows)
    family_counts = Counter(str(row["db_name"]) for row in rows)
    language_counts = Counter(str(row["bull_language"]) for row in rows)
    exact_rows = [row for row in phase2_rows if row["match_type"] == "exact"]
    high_sim_rows = [row for row in phase2_rows if row["match_type"] == "similar"]

    lines = [
        "# BULL Output Column Hints",
        "",
        "This note mines public BULL-cn train/dev questions and structured SELECT",
        "lists to identify generic output-column priors for Phase 2 finance tasks.",
        "It does not use hidden test labels or task-specific hard-coding.",
        "",
        "## Source",
        "",
        f"- BULL rows analyzed: {total}",
        "- Files: `/tmp/phase2_source_check/BULL_full/BULL/BULL-cn/{train,dev}.json`, `BULL-en/{train,dev}.json`",
        f"- Languages: {', '.join(f'{k}={v}' for k, v in sorted(language_counts.items()))}",
        f"- Families: {', '.join(f'{k}={v}' for k, v in sorted(family_counts.items()))}",
        f"- Phase 2 exact question matches: {len(exact_rows)} / {len(phase2_rows)}",
        f"- Phase 2 high-similarity non-exact matches (ratio >= 0.72): {len(high_sim_rows)}",
        "",
        "## Strong Learned Hints",
        "",
        "1. **Do not auto-add date/period columns for data/record wording.**",
        "   In BULL, many questions that say data/records still output only the requested metric.",
        "   Example: `我国第三产业国内生产总值这些年来的记录是什么样的` selects only `thirdindustrygdp`.",
        "2. **Distribution questions usually need the group label plus the requested aggregate.**",
        "   For `最高学历分布情况`, BULL uses `education, COUNT(*)`, not a broad manager table.",
        "3. **Return-rate data questions are often metric-only.**",
        "   `十年回报率数据` selects `rrintenyear` only; fund name/date/code are not output unless asked.",
        "4. **Chinese finance entity outputs prefer display descriptors.**",
        "   Stock/company/fund questions commonly output `ChiNameAbbr`, `AShareAbbr`, or `SecuAbbr`; codes appear when code wording is explicit.",
        "5. **Context/filter columns are not output columns.**",
        "   Columns used to filter by year, index cycle, threshold, region, or entity are usually absent from SELECT unless explicitly requested.",
        "",
        "## Cue Summary",
        "",
        "| cue | n | agg rate | count rate | date output rate | id/code output rate | display-name output rate | top outputs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in cue_rows:
        lines.append(
            "| {cue} | {n} | {agg_rate} | {count_rate} | {date_rate} | {id_rate} | {display_rate} | {top_outputs} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Phase 2 Exact Matches",
            "",
            "| task | family | question | BULL SELECT | gold header |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in exact_rows:
        lines.append(
            "| {task_id} | {db_name} | {question_short} | `{select_columns}` | `{gold_header}` |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Top Output Columns By Family",
            "",
            "| family | output | n |",
            "| --- | --- | ---: |",
        ]
    )
    for row in top_column_rows[:45]:
        lines.append(f"| {row['db_name']} | `{row['select_column']}` | {row['n']} |")

    lines.extend(
        [
            "",
            "## Suggested Prompt Seed",
            "",
            "```text",
            "Use public BULL-style output-shape priors only as generic guidance:",
            "- Return only columns explicitly requested as final answer fields.",
            "- Do not include filter, sort, threshold, period, date, id, or evidence columns unless the question asks to output them.",
            "- For data/record questions, output the requested value/metric columns; do not add dates just because rows vary over time.",
            "- For distribution/grouped count questions, output the group label and COUNT(*).",
            "- In Chinese finance schemas, prefer display/short-name columns for entities unless code or full legal name is requested.",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine BULL output-column priors for Phase 2 finance tasks.")
    parser.add_argument(
        "--bull-root",
        type=Path,
        action="append",
        default=None,
        help="BULL language root; repeat to use multiple roots. Defaults to BULL-cn and BULL-en.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "artifacts" / "bull_output_column_hints",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.72)
    args = parser.parse_args()

    bull_roots = args.bull_root or DEFAULT_BULL_ROOTS
    bull_rows = _load_bull_rows(bull_roots)
    if not bull_rows:
        raise SystemExit(f"No BULL rows found under {bull_roots}")

    select_rows: list[dict[str, Any]] = []
    cue_counters: dict[str, Counter[str]] = defaultdict(Counter)
    cue_output_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_output_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in bull_rows:
        items = _select_items(row)
        rendered = [item.rendered for item in items]
        row_cues = _cues(str(row["question"]))
        family = str(row["db_name"])
        language = str(row.get("_bull_language", ""))
        has_agg = any(item.agg for item in items)
        has_count = any(item.agg.upper() == "COUNT" for item in items)
        has_date = _has_date_output(items)
        has_id = _has_id_output(items)
        has_display = _has_display_output(items)

        for item in items:
            family_output_counts[family][item.rendered] += 1
        for cue in row_cues:
            cue_counters[cue]["n"] += 1
            cue_counters[cue]["has_agg"] += int(has_agg)
            cue_counters[cue]["has_count"] += int(has_count)
            cue_counters[cue]["has_date"] += int(has_date)
            cue_counters[cue]["has_id"] += int(has_id)
            cue_counters[cue]["has_display"] += int(has_display)
            cue_output_counts[cue].update(rendered)

        select_rows.append(
            {
                "q_id": row.get("q_id"),
                "bull_language": language,
                "db_name": family,
                "question": row["question"],
                "cues": ",".join(row_cues),
                "kind": _question_kind(items),
                "select_columns": " | ".join(rendered),
                "has_agg": has_agg,
                "has_count": has_count,
                "has_date_output": has_date,
                "has_id_or_code_output": has_id,
                "has_display_output": has_display,
                "sql_query": row.get("sql_query", ""),
                "bull_file": row.get("_bull_file", ""),
            }
        )

    cue_rows: list[dict[str, Any]] = []
    for cue, counts in sorted(cue_counters.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        n = counts["n"]
        cue_rows.append(
            {
                "cue": cue,
                "n": n,
                "agg_rate": _pct(counts["has_agg"], n),
                "count_rate": _pct(counts["has_count"], n),
                "date_rate": _pct(counts["has_date"], n),
                "id_rate": _pct(counts["has_id"], n),
                "display_rate": _pct(counts["has_display"], n),
                "top_outputs": "; ".join(f"{col} ({cnt})" for col, cnt in cue_output_counts[cue].most_common(8)),
            }
        )

    top_column_rows: list[dict[str, Any]] = []
    for family, counter in sorted(family_output_counts.items()):
        for column, count in counter.most_common(20):
            top_column_rows.append({"db_name": family, "select_column": column, "n": count})

    phase2_rows: list[dict[str, Any]] = []
    exact_by_question = {str(row["question"]): row for row in bull_rows}
    for task in _load_phase2_questions():
        task_id = task["task_id"]
        question = task["question"]
        exact = exact_by_question.get(question)
        best_score, best = _best_bull_match(question, bull_rows)
        match_type = "none"
        match_row: dict[str, Any] | None = None
        if exact is not None:
            match_type = "exact"
            match_row = exact
        elif best is not None and best_score >= args.similarity_threshold:
            match_type = "similar"
            match_row = best

        items = _select_items(match_row) if match_row else []
        phase2_rows.append(
            {
                "task_id": task_id,
                "question": question,
                "question_short": question.replace("|", "/")[:90],
                "match_type": match_type,
                "similarity": f"{best_score:.3f}" if best else "",
                "db_name": str(match_row["db_name"]) if match_row else "",
                "bull_language": str(match_row.get("_bull_language", "")) if match_row else "",
                "bull_question": str(match_row["question"]) if match_row else "",
                "select_columns": " | ".join(item.rendered for item in items),
                "gold_header": " | ".join(_read_gold_header(task_id)),
                "sql_query": str(match_row.get("sql_query", "")) if match_row else "",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        args.out_dir / "bull_select_examples.csv",
        select_rows,
        [
            "q_id",
            "bull_language",
            "db_name",
            "question",
            "cues",
            "kind",
            "select_columns",
            "has_agg",
            "has_count",
            "has_date_output",
            "has_id_or_code_output",
            "has_display_output",
            "sql_query",
            "bull_file",
        ],
    )
    _write_csv(
        args.out_dir / "cue_summary.csv",
        cue_rows,
        ["cue", "n", "agg_rate", "count_rate", "date_rate", "id_rate", "display_rate", "top_outputs"],
    )
    _write_csv(args.out_dir / "top_columns_by_family.csv", top_column_rows, ["db_name", "select_column", "n"])
    _write_csv(
        args.out_dir / "phase2_bull_matches.csv",
        phase2_rows,
        [
            "task_id",
            "question",
            "question_short",
            "match_type",
            "similarity",
            "db_name",
            "bull_language",
            "bull_question",
            "select_columns",
            "gold_header",
            "sql_query",
        ],
    )

    (args.out_dir / "bull_output_column_hints.md").write_text(
        _make_markdown(
            rows=select_rows,
            cue_rows=cue_rows,
            phase2_rows=phase2_rows,
            top_column_rows=top_column_rows,
        )
        + "\n",
        encoding="utf-8",
    )

    exact_count = sum(1 for row in phase2_rows if row["match_type"] == "exact")
    similar_count = sum(1 for row in phase2_rows if row["match_type"] == "similar")
    print(f"BULL rows: {len(bull_rows)}")
    print(f"Phase2 exact matches: {exact_count}")
    print(f"Phase2 similar non-exact matches: {similar_count}")
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
