# DataAgent-Bench — Data Analysis Report

> 🌐 **Language**: **English** · [한국어](DATA_ANALYSIS.ko.md) · [中文](DATA_ANALYSIS.zh.md)

A quantitative analysis of the public 50-task set for the KDD Cup 2026 DataAgent-Bench challenge, plus the v2 (2026-04-29) holdout measurements. Forms the basis for deciding what to change in v3+.

> Scope of this doc: **the dataset itself + how our agent behaves on top of it**.
> System internals live in `[ARCHITECTURE.md](ARCHITECTURE.md)`; submission history and score trends in `[SUBMISSION_LOG.md](SUBMISSION_LOG.md)`; operating manual in `[../CLAUDE.md](../CLAUDE.md)`.

---

## 1. At a glance


| Item                    | Value                                                                    |
| ----------------------- | ------------------------------------------------------------------------ |
| Public tasks            | 50                                                                       |
| Splits                  | train 40 / holdout 10 / smoke 5 (deterministic via `scoring/holdout.py`) |
| Difficulty distribution | easy 15 / medium 23 / hard 11 / extreme 1                                |
| Context file types      | `.csv`, `.json`, `.db` (SQLite), `.md` only — no PDF/Excel/Parquet/image |
| Always-present asset    | `context/knowledge.md` (50/50 — domain dictionary, table descriptions)   |
| Largest context         | task_257 = **441 MB** (medium)                                           |
| Typical answer shape    | 1–3 columns, median 1 row, max 140                                       |


Headline implications:

- Every public task ships `knowledge.md` — failure to auto-inject loses domain context (already addressed in v2).
- Most ground truths are 1×1 cells (lookup-style) — a single normalization mismatch zeroes the task.
- Phase 1 data has no PDF/Excel/image, but the organizers' hidden set is explicitly multi-modal → reader candidates beyond v3.

---

## 2. Difficulty × context shape (50 tasks)

### 2.1 Difficulty distribution


| Difficulty | n   | share | rule definition                                |
| ---------- | --- | ----- | ---------------------------------------------- |
| easy       | 15  | 30%   | structured files + knowledge doc               |
| medium     | 23  | 46%   | structured + DB + docs                         |
| hard       | 11  | 22%   | multi-source + unstructured (~10K–128K tokens) |
| extreme    | 1   | 2%    | ultra-long inputs (>128K tokens)               |


Only one extreme task here, but the hidden set may have more — runner's `max_steps_by_difficulty` mapping (extreme=28) is sized for that scenario.

### 2.2 Context subtree frequency (out of 50)


| Subtree              | Count  | Purpose                              |
| -------------------- | ------ | ------------------------------------ |
| `csv/`               | 36     | tabular                              |
| `json/`              | 30     | structured (often nested objects)    |
| `db/` (SQLite `.db`) | 27     | normalized tables — JOIN/aggregation |
| `doc/`               | 12     | supplementary docs (`.md`)           |
| `**knowledge.md`**   | **50** | domain dictionary — 100% present     |


Combination patterns (top-frequency in 50):

- CSV+JSON (12) — foreign-key lookup
- CSV+DB (10) — DB schema → CSV adds facts
- JSON+DB (6) — DB normalized, JSON nested
- CSV+JSON+DB (4) — three-source combination (mostly medium+)

### 2.3 File extension totals


| Extension | Count | Note                       |
| --------- | ----- | -------------------------- |
| `.md`     | 64    | knowledge.md 50 + doc/* 14 |
| `.csv`    | 40    |                            |
| `.json`   | 37    |                            |
| `.db`     | 27    | SQLite                     |
| Others    | **0** | no PDF/Excel/Parquet/image |


---

## 3. Context-size distribution — memory-load indicator


| Difficulty | n   | min    | median | max        |
| ---------- | --- | ------ | ------ | ---------- |
| easy       | 15  | 17 KB  | 287 KB | **58 MB**  |
| medium     | 23  | 38 KB  | 1.4 MB | **441 MB** |
| hard       | 11  | 41 KB  | 256 KB | **267 MB** |
| extreme    | 1   | 376 KB | 376 KB | 376 KB     |


**Eight large contexts (≥50 MB):**


| task_id  | difficulty | size   |
| -------- | ---------- | ------ |
| task_257 | medium     | 441 MB |
| task_250 | medium     | 384 MB |
| task_330 | hard       | 267 MB |
| task_259 | medium     | 182 MB |
| task_249 | medium     | 166 MB |
| task_243 | medium     | 137 MB |
| task_420 | hard       | 59 MB  |
| task_38  | easy       | 58 MB  |


**Implications:**

- 32K context window (our vLLM) cannot ingest the raw payload. → `dataframe_describe`/`dataframe_head` compression prepass is mandatory (introduced in v2).
- A reload-per-call IO pattern hits the 30s limit. → persistent IPython kernel (introduced in v2) shows direct ROI on task_249 / task_250-class large cases.

---

## 4. Ground-truth (`gold.csv`) shape distribution


| Shape     | Count         |
| --------- | ------------- |
| 1 column  | 40 / 50 (80%) |
| 2 columns | 7 / 50        |
| 3 columns | 3 / 50        |


Row count: min 1, **median 1**, p90 7, max 140.

**Implications:**

- 1×1 single-value answers dominate → easy tier is mostly lookup-style questions.
- Column ablation (Phase 3 plan) only fires at 6+ columns; on the public set it almost never triggers → keep the code in for hidden-set wide-table cases.
- A 140-row long-list also exists (task_180) — for long-list answers normalization cost grows linearly.

---

## 5. Question (natural-language) length


| Statistic | Value     |
| --------- | --------- |
| min       | 30 chars  |
| median    | 90 chars  |
| max       | 144 chars |
| mean      | 88 chars  |


Questions are short — the model has to infer intent across a wide gap. The system prompt must spell out answer formatting (normalization, column signature) to minimize score loss (introduced in v2).

---

## 6. v2 holdout measurement (2026-04-29)

### 6.1 Score summary (`λ ∈ {0.05, 0.10, 0.20}`)


| λ        | mean Score | mean Recall |
| -------- | ---------- | ----------- |
| 0.05     | 0.6300     | 0.6333      |
| **0.10** | **0.6267** | **0.6333**  |
| 0.20     | 0.6200     | 0.6333      |


Score is nearly flat across the three λ — the extra-column penalty has little effect (we don't over-emit columns).

### 6.2 By difficulty (container, λ=0.10)


| difficulty | n   | score      | recall |
| ---------- | --- | ---------- | ------ |
| easy       | 3   | 0.6667     | 0.6667 |
| medium     | 5   | **0.8000** | 0.8000 |
| hard       | 2   | **0.1333** | 0.1667 |
| extreme    | 0   | —          | —      |


**Reading:**

- **medium 0.80** — knowledge.md injection + dataframe prepass + persistent kernel together hit their stride. This bucket is hard to extract more ROI from in v3 (near plateau).
- **easy 0.67** — 1×1 lookups, but 1/3 still scored 0. Suspect normalization or over-emission.
- **hard 0.13** — biggest leak. Primary v3 target.

### 6.3 holdout 10-task per-row measurement

Per-task context shape + answer-match pattern:


| task_id  | diff   | ctx subtrees | size       | steps | gold (c×r) | pred (c×r) | shape match | verdict                                                 |
| -------- | ------ | ------------ | ---------- | ----- | ---------- | ---------- | ----------- | ------------------------------------------------------- |
| task_11  | easy   | json         | 0.5 MB     | 5     | 3 × 3      | 3 × 6      | **rows ↑**  | column signature broken (over-emission)                 |
| task_27  | easy   | json         | 24 KB      | 5     | 3 × 1      | 3 × 1      | OK          | likely matched                                          |
| task_75  | easy   | csv+json     | 0.5 MB     | 8     | 1 × 1      | 1 × 1      | OK          | likely matched                                          |
| task_169 | medium | csv+db       | 9 MB       | 7     | 1 × 1      | 1 × 1      | OK          | matched                                                 |
| task_249 | medium | db+json      | 174 MB     | 7     | 2 × 1      | 2 × 1      | OK          | matched (large context OK)                              |
| task_250 | medium | csv+db+json  | **403 MB** | 6     | 1 × 1      | 1 × 1      | OK          | matched (kernel/dataframe payoff)                       |
| task_261 | medium | csv+db+json  | 0.13 MB    | 6     | 1 × 1      | 1 × 1      | OK          | matched                                                 |
| task_303 | medium | db+json      | 0.24 MB    | 9     | 1 × 1      | 1 × 1      | OK          | matched                                                 |
| task_355 | hard   | csv+doc      | 41 KB      | 6     | **3 × 1**  | 2 × 1      | **cols ↓**  | column missing (under-emission)                         |
| task_408 | hard   | db+doc       | 1.1 MB     | 8     | 1 × 1      | 1 × 1      | OK          | value mismatch (normalization / wrong answer) suspected |


**All 10 tasks reach `_answer`** (succeeded=True) — i.e. recovered from 0/10 to 10/10. The remaining score leak is "reached but answer wrong".

---

## 7. Failure pattern taxonomy

Three modes separate cleanly when looking at v2 holdout zero / partial-score tasks.

### 7.1 Over-emission (rows ↑) — task_11

`gold` 3×3, our `pred` 3×6. Same 3 columns but 6 rows. The scorer builds a sorted-value multiset signature after normalization, so extra rows mismatch the multiset → 0 columns matched. We essentially **submitted 3 correct rows + 6 extra ones**. This single task drags down the easy-tier average.

**Mitigations:**

- Add a system-prompt cue: "Include only the exact row count the question asks for; no extras."
- `answer_validator` (Phase 3) warns when the agent returns raw query results without dedup/filter even though gold rows are unknown.

### 7.2 Under-emission (cols ↓) — task_355

`gold` 3×1 vs `pred` 2×1. One column dropped. Scorer matches at most 2 → recall ≤ 2/3, λ-penalty 0. One hard-tier task scores 0–0.33.

**Mitigations:**

- "When in doubt about including a column, INCLUDE" was already in v2's prompt, but on hard tasks the model still drops one column under multi-source joins.
- `answer_validator` heuristic: count noun-phrases in the question (e.g. "list their A, B, and C") vs predicted columns.

### 7.3 Value mismatch (normalization / computation) — task_408

`gold` 1×1, `pred` 1×1, shape matches but score is 0 (or partial). The hardest mode: shape is right but the value differs. Likely causes:

- Wrong SQL/Python computation result
- Normalization — numeric ROUND_HALF_UP vs HALF_EVEN, date format, missed trim
- Misinterpretation of unstructured cues in `doc/`

**Mitigations:**

- Direct trace.json analysis to find the wrong-inference step — task_408 trace is 8 steps and marathon-style reasoning. Likely a wrong domain assumption.
- Add ROUND_HALF_UP normalization preview for numeric columns in `dataframe_describe` output.

---

## 8. Strengths / weaknesses one-pager

### Strengths (as of v2)

1. **Medium 0.80** — knowledge.md auto-inject + persistent kernel + dataframe_describe form a synergy.
2. **All large-context (≥100 MB) tasks pass** — task_249 (174 MB), task_250 (403 MB) both matched. Validates persistent kernel.
3. **Easy + medium 1×1 lookups** — JSON-mode probe + plan-then-execute prompt are stable on single-value extraction.
4. **All tasks reach `_answer`** — 10/10. After the `_coerce_action_input` hotfix there's no step waste.

### Weaknesses (v3+ targets)


| Priority | Area                 | Evidence                                      | Candidate                                       |
| -------- | -------------------- | --------------------------------------------- | ----------------------------------------------- |
| 🔴 High  | hard 0.13            | task_355 col missing, task_408 value mismatch | answer_validator + question-parsing heuristic   |
| 🟠 Mid   | easy 1/3 zero        | task_11 over-emission                         | "exact row count" prompt emphasis               |
| 🟡 Low   | hidden-set PDF/Excel | 0 in public set but rules mention multi-modal | add readers per Phase 1.5 plan                  |
| 🟡 Low   | extreme (>128K)      | 1 in public, only 376 KB                      | YaRN/RoPE extension stays vLLM's responsibility |


---

## 9. Next-round (v3) candidates — ROI ordered

1. **answer_validator + conditional terminal** (Phase 3) — before `_answer`: (a) check numeric column dtype consistency, (b) compare question noun-phrase count vs answer column count, (c) warn on suspicious cells. Warnings come back as observations → the agent fixes and re-submits.
2. **Hard-tier trace debug** — beyond task_355 / task_408, analyze the 11 hard traces in train 40. Find common failure modes, add 1–2 lines to the system prompt.
3. **Column-signature self-consistency** (Phase 4 gate) — k=3, temperature 0→0.5, multiset voting. Absorb stochastic wrong answers on hard tasks via majority. Only ship if 12h budget simulation passes.
4. **Extra-row guard** — alongside `_coerce_action_input`, a "rows abnormally long" heuristic warning (with a per-task conservative cap when gold is unknown).

After each change, re-measure on the 10-task holdout → only ship v3 if mean Score ≥ v2 + 3 points.

---

## 10. Data refresh procedure

§2–§5 are **static analysis of the public 50-task set** — stable as long as the dataset doesn't change. §6–§8 must be **refreshed each submission**:

```bash
# 1. New run
uv run dabench run-benchmark --config configs/local.yaml \
  --task-set data/public/holdout_ids.txt

# 2. mock_scorer
uv run python -m data_agent_baseline.scoring.mock_scorer \
  --predictions artifacts/runs/<run_id> \
  --gold data/public/output --input data/public/input \
  --lambda-values 0.05 0.10 0.20

# 3. per-task shape extract (for §6.3 table refresh)
.venv/bin/python -c "
import json, pathlib
run_dir = pathlib.Path('artifacts/runs/<run_id>')
for trace in sorted(run_dir.glob('task_*/trace.json')):
    d = json.loads(trace.read_text())
    print(d['task_id'], d.get('succeeded'), len(d.get('steps', [])))
"
```

The repo `.gitignore` whitelists this doc (`!docs/DATA_ANALYSIS.md`) for git tracking — for team sharing.