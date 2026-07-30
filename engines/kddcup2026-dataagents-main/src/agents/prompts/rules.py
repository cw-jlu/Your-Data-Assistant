"""Shared prompt constants and data-analysis rules."""

from __future__ import annotations

# 所有工具的 `path` 字段共用同一条路径约定：
# - 必须相对于任务的 context 目录
# - **不要** 带 `context/` 前缀，哪怕 system prompt 里提到过 `context/`
# 用常量收敛文案，后续调整只改一处，保证路径字段描述一致，减少模型的歧义。
PATH_CONVENTION_NOTE = (
    "Paths are relative to the task context directory; DO NOT prepend 'context/'. "
    "For example use 'knowledge.md' or 'csv/foo.csv', not 'context/knowledge.md'."
)

# 任务框架：接收什么（question + context/）→ 怎么开始（explore）→
# explore 返回什么（schemas / PDF entity groups / knowledge / join_paths / unit_conversion …）→
# 怎么提交（answer tool）→ 访问边界（只读 context/）
TASK_PROFILE = (
    "For each task you receive one question (may be Chinese) and a `context/` "
    "directory containing any subset of: `csv/`, `db/` (SQLite), `json/`, "
    "`doc/` (.md/.pdf), `video/` (MP4) — not all types are present in every task. "
    "Call `explore` first — it returns a complete data map: schemas, "
    "PDF entity group samples (`entity_groups_sample`) for `doc/*.pdf`, the "
    "`knowledge` field (field definitions, formulas, value codes from "
    "`knowledge.md`), prose documents that require `run_etl`, join paths, "
    "value samples, warnings, and "
    "`unit_conversion`. Use these findings as your primary context for all "
    "subsequent computation. "
    "Video content reaches you only as `video_findings` in the `explore` result; "
    "raw video is never attached to the conversation. "
    "Submit the final answer as a structured table via the `answer` tool. "
    "You may only read files inside that `context/` directory."
)

# 共享：answer 表的内容契约——schema mirroring / WH-target / cell type
DATA_OUTPUT_RULES = """\
- Unit conversion: when `explore` returns a `unit_conversion` array, each entry gives the exact columns and multiplication factor needed. Apply all conversions in `execute_python` before submitting — unconverted values score 0. Conversion factors follow the semantic field: when answering from a synonym/substitute column, ALWAYS apply the named column's factor to it.
- Default unit: when the `knowledge` field from `explore` declares a default storage unit (e.g. "百万元", "万元") AND `explore` did NOT return a `unit_conversion` for those fields, the data values are ALREADY in that unit. Do NOT convert them to 元 or any other unit unless the question explicitly requests a different unit.
- Unit suffixes in numeric measure cells: when a result column is a numeric measure, column names MAY contain unit indicators like `(%)`, `(万元)`, `(亿)`, but numeric cell values MUST be bare numbers — NEVER append `%`, `万`, `亿`, `元`, `USD`, `kg`, or any other unit suffix to a numeric value cell. Strip suffixes and convert to plain numeric before submitting. Example: column `收益率(%)` with value `48.706` is correct; `48.706%` is wrong and will not match the gold reference. This does NOT apply to text/code/unit-label columns the question asks you to return; preserve those string values exactly.
- Mirror the source schema: NEVER concatenate, split, or rename source columns. Keep them exactly as stored (e.g., separate `first_name`/`last_name` stay separate; CJK names like 谢理斌 stay as one column). Only synthesize a derived column when the question explicitly requires a computed value.
- Data-listing trigger: questions that ask to 查/看/找/搜/显示/展示/列出/retrieve/show/check named fields as 数据/记录/明细/情况/是多少 are data-listing tasks. Return source rows at original grain for those fields; do NOT require every requested column to be non-null unless the question explicitly says 非空/不为空/有数据/non-null/not empty.
- Data-listing answers: return ALL source rows as-is. NEVER aggregate (GROUP BY + SUM/AVG/COUNT) unless the question contains an explicit aggregation keyword ("total"/"sum"/"average"/"count"/"how many"/"统计"/"总计"/"平均"/"汇总"/"一共"/"总共"/"数目"/"数量"). NEVER truncate with LIMIT/`head()`/`iloc[:N]` unless the question contains a superlative ("最大"/"最小"/"最新"/"最高"/"最低"/"top N"/"highest"/"lowest"). NEVER filter out NULL rows (`IS NOT NULL`/`dropna()`/`notna()`) UNLESS the question explicitly asks for non-null records ("非空数据"/"非空记录"/"不为空"/"non-null"). NULL rows are part of the source; the grader matches row counts exactly. "我国"/"咱们" scopes the dataset, it does NOT request aggregation — when no national-total row exists, return all per-province rows.
- Entity name: when multiple name columns exist for the same entity, ALWAYS output the SHORTEST one unless the question explicitly says "full name"/"全称"/"完整名称".
- Fund name column priority (shortest wins): three tiers exist — (1) `ChiName`/`fund_name` full name, (2) `ChiNameAbbr`/`fund_name_short` market name, (3) `SecuAbbr`/`secuabbr` trading name (shortest). ALWAYS output the SHORTEST available: prefer `SecuAbbr`/`secuabbr`; fall back to `ChiNameAbbr`/`fund_name_short` only when `SecuAbbr`/`secuabbr` is absent; NEVER output `ChiName`/`fund_name` unless the question explicitly says "full name"/"全称"/"完整名称". Even after a JOIN, select the shortest name column already present — do NOT substitute a longer name from the joined table.
  Example: question asks "基金简称" and both `ChiNameAbbr` and `SecuAbbr` exist → output `SecuAbbr` (shortest).
- Preserve cell types from the source. Submit integer-valued IDs as JSON integers (`163109`), not as floats (`163109.0`); submit dates and codes as strings exactly as they appear in the source.
- "Current year"/"today": use `date.today()` in `execute_python`, NOT the latest date in the dataset.
- Preserve full numeric precision for computed values (averages, ratios, sums, etc.). Submit the raw value as Python prints it (e.g. `60.77956989247312`), not a human-readable rounded form (e.g. `60.78`). Round only when the question explicitly asks for a specific precision.
- Division-by-zero: pandas produces `inf` which passes `> X` comparisons, silently injecting spurious rows. Exclude zero-denominator rows before division or replace inf with NaN. Prefer filtering in SQL when source is SQLite.
- pandas merge/join: before merging, keep only join keys and columns needed later; do not rely on default `_x`/`_y` for non-key overlaps — rename semantically if both sides are needed.
- Artifact handoff: >10 rows OR >50 cells → MUST write CSV, do NOT inline `columns`/`rows`. In `execute_python`: (1) project to WH-target columns only — `answer_df = filtered[['col_a', 'col_b']]` then `answer_df.to_csv(out_path, index=False)`. NEVER `df.to_csv(...)` on the whole frame; NEVER use `csv.DictWriter` or string concat (full-width commas `，` corrupt headers). (2) Print row count, sample rows, and column list. (3) `out_path = os.path.join(os.environ['DABENCH_ANSWER_DIR'], 'answer.csv')` — ALWAYS use this env var, NEVER hardcode paths like `/tmp/...`. Write CSV then `print(out_path)`. (4) `answer({"from_csv": "<the printed path>"})` — copy path from print output, NEVER guess. Inline `columns`/`rows` is default for ≤10 rows AND ≤50 cells.
- Always print a verification snapshot (row count + sample rows + final column list) from `execute_python` BEFORE calling `answer`. NEVER print the entire DataFrame — use `head()` or slicing for the snapshot. Re-read the question: if the column count exceeds the number of attributes the question explicitly asks you to output, you are leaking source columns — trim to just the WH-target; the grader penalizes extra columns. When the question asks "what is the comment/post/answer/message/title/name", verify the snapshot shows the text body column (Text/Body/Title/Name), NOT an Id/Code column. The snapshot must show FULL cell values: call `pd.set_option('display.max_colwidth', None)` before printing; if a value ends with `...` it is truncated — re-query the full value before submitting, NEVER reconstruct or guess.
- Single-value questions get single-column answers. For questions asking "how many times", "what percentage", "what is the average/ratio/count", the answer is ONE computed value in ONE column. Intermediate computation columns (counts, subtotals, IDs used for filtering) must NOT appear in the answer — they are working variables, not results.
- Ungrouped aggregates: ALWAYS submit ONE row with one column per aggregate (`SELECT MAX(a), MIN(a), MAX(b), MIN(b)` → 1 row × 4 columns). NEVER pivot aggregates into rows, NEVER add a metric-label column (`指标`/`统计项`/`metric`/`statistic`), NEVER transpose into a label+value layout. GROUP BY results keep one row per group. When a use-case SQL in the `knowledge` field matches the question, mirror its result shape exactly.
- DISTINCT: ALWAYS use `SELECT DISTINCT` / `drop_duplicates()` when the question asks "which/who/what entities" (people, shareholders, teams, schools, companies). An entity may appear in multiple rows of the source table (e.g., one shareholder across multiple stocks); the answer must list each entity ONCE. Do NOT add DISTINCT when asking about records/entries (transactions, log entries) — preserve every qualifying row. Keywords "distinct"/"unique"/"different" in the question always force DISTINCT. When listing distinct values, output one column of deduplicated values only — do NOT add a frequency/count column alongside. Example: "which shareholders qualify" → `SELECT DISTINCT 股东名称 FROM …`; "list all transactions above 1M" → no DISTINCT.
- Superlative tie-breaking: use `ORDER BY <col> DESC/ASC LIMIT 1` only. NEVER add a secondary sort key or treat record IDs as time — the gold reference relies on natural storage order for ties.
- JOIN discipline: NEVER join to fetch a column the base table already has. Before any `pd.merge`/`JOIN`, cast both join keys to the same dtype (`astype(str)` in pandas, `CAST(col AS TEXT)` in SQL) — mismatched int vs str silently drops rows or raises `MergeError`.
- Path discipline: `execute_python` CWD is the context directory, `context_root` is pre-injected. Use relative paths or `os.path.join(context_root, ...)`. NEVER hard-code absolute paths. Answer artifact directory: `os.environ['DABENCH_ANSWER_DIR']`.
- "No X" / "does not have X" filters: check whether the data uses NULL/empty values or absent rows, then pick `WHERE col IS NULL OR col = ''` vs `NOT EXISTS` accordingly. NEVER use `NOT IN` on a subquery that may contain NULLs.
- Zero is a valid data value, NOT a null sentinel. Do NOT add `> 0` or `!= 0` filters unless the `knowledge` field explicitly says 0 means "missing"."""

# 共享：path 字段约定段。`PATH_CONVENTION_NOTE` 在本模块定义，
# 被 5 个 path 字段的 schema description（tools/inputs.py）复用；
# 这里再插一份到 system prompt 作远端兜底
# （schema description 不会被渲染进 prompt）。
PATH_CONVENTION_BLOCK = (
    "Path convention (applies to every tool `path` argument):\n" + PATH_CONVENTION_NOTE
)

# 共享：歧义检测与消解规则。自然语言问题到 SQL/pandas 的映射过程中
# 常见四类歧义（AmbiSchema / AmbiValue / AmbiVague / 数值引用歧义），
# 传统做法靠交互澄清，本 agent 无法交互，需在 DISCOVER/VERIFY 阶段
# 通过 schema 采样 + explore knowledge 字段 + 多候选验证自行消解。
DISAMBIGUATION_RULES = """\
- Ambiguity resolution — always resolve via `SELECT DISTINCT col` + the `knowledge` field from `explore`, never guess:
  (a) Schema: phrase maps to multiple columns ("Fresno" → `City` vs `County`) → check which column's values contain the phrase.
  (b) Value: term absent or non-standard in data ("Vietnam War end" → 1973 vs 1975) → search actual values, check `knowledge` code mappings.
  (c) Vague intent: metric unspecified ("largest city" → area vs population) → prefer the most conventional interpretation, do not stall.
  (d) Numeric ref: value matches multiple columns → sample value ranges to pick the right column.
- Column-first resolution: when the question hinges on an attribute (metric, date, status) rather than a table entity, identify the target column first, then trace back to its table and join path. Column names carry richer semantic signals than generic table names.
- The `knowledge` field from `explore` is the primary disambiguation source. It typically maps business terms to exact column names, value codes, and formulas. Consult it BEFORE attempting any filter that involves domain-specific terms.
- When multiple SQL interpretations are valid, prefer the one that returns a non-empty result. Empty result from an ambiguous filter → re-examine column/value choice.
- "How many times is A compared to B" = RATIO (`A / B`), NOT a count. Only interpret as count when describing a repeatable event.
- Compound filters: verify each filter independently with `SELECT DISTINCT col` before combining. If any filter yields zero results, re-examine the column mapping.
- "Not yet X" / "haven't reached X" means `< X`, NOT `!= X`.
- Format tolerance: when the question's value format differs from the data's (e.g., question `0:01:54` vs data `1:54.455`; question `$5,000` vs data `5000.00`; question `2024-01` vs data `2024-01-15`), normalize or truncate to the coarsest shared form and match on equivalence — do NOT insist on string-literal equality, do NOT fall back to "closest numeric match".
- Categorical filters: use EXACT equality (`= 'X'`), NOT `LIKE '%X%'`. Always `SELECT DISTINCT col` first to find the exact value. Only use LIKE when the question says "contains"/"includes".
- Date/time format: `strftime`/`julianday` silently return NULL for non-ISO dates, dropping rows without error. ALWAYS run `SELECT col FROM tbl WHERE col IS NOT NULL LIMIT 3` to check the actual format before any date operation. If not `YYYY-MM-DD`, use `SUBSTR` or parse in `execute_python` with `pd.to_datetime(..., format=...)`. Column name is NOT evidence of format.
"""
