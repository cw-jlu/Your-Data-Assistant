# exp_110 Plan-Solve-Verify: 詳細実装プラン

> 路線 A: 思考の戦略的配分で **+0.03〜+0.06 local 改善 → LB 0.52-0.54 (= 4 位確保 + 3 位射程)** を狙う。
> ベース: `exp_109_plan_first_strengthen` (= 現 best 同等 0.7328 ± 0.0065)。
> SOTA 系 (CHESS / MAC-SQL) の **3-stage pipeline 簡易版**。

---

## 1. 設計概要

### 3 phase 構造

```
[Phase 1] Question Understanding + Plan generation
  Input:  task.question + rich preamble
  Action: 1 LLM call (no tools)
  Thinking budget: extended (~16K tokens)
  Output: structured plan JSON

[Phase 2] ReAct execution (= 現 exp_109、ただし plan 注入済)
  Input:  question + preamble + Phase 1 plan
  Action: 通常 ReAct loop (max 24 step、現 32 から削減)
  Thinking: per step ~8K tokens (= step が plan で focused)
  Output: candidate answer

[Phase 3] Verification + targeted repair
  Input:  Phase 1 plan + Phase 2 answer
  Action: 1 LLM call (no tools), extended thinking
  Thinking budget: ~8K tokens
  Output: verdict + optional fix instruction

[Phase 4] Optional repair (if Phase 3 flags issue)
  Input:  Phase 3 fix instruction + Phase 2 answer
  Action: 1 ReAct turn (= max 4 step) for targeted fix
  Output: final answer

→ commit final answer to prediction.csv
```

### 何を解決するか (= 50-task 分析の Type 別救済)

| Failure Type | Phase で対処 |
|---|---|
| **A 列スコープ違反** | Phase 1 で column_count 厳格 commit + Phase 3 で verify |
| **B 計算ロジック誤り** (= avg/12 抜け) | Phase 1 で `calculation_outline` 明示 + Phase 3 で magnitude sanity check |
| **C Filter scope** | Phase 1 で `expected_row_range` 仮説 + Phase 3 で row count vs scope 比較 |
| **D Tied-rows** | Phase 1 で `row_count_category="tied"` 明示 + Rule 10 filter-back 維持 |
| **E 質問解釈 bias** | Phase 1 で `interpretations` (主 + 代替) 列挙、執行は主の解釈で実行 |
| **F Doc narrative** | Phase 1 で **どの doc を読むべきか** + 必要な情報リスト明示 |

→ **Type B/C/E が直接攻略**。Type A は exp_109 既存 Rule 18 を引き継ぎ。Type G (= medical 閾値) は本案で攻略困難 → 路線 C で別途。

### 何を解決しないか (= 既存 exp_109 から引き継ぐ強み)

- ✅ Rich preamble (= per-col profile + raw)
- ✅ Padding fix (= mode-by-attempts length group)
- ✅ row_count predeclaration 撤廃
- ✅ Rule 18 column_count VERIFY
- ✅ Rule 10 filter-back for superlatives

→ exp_110 = exp_109 を Phase 2 にして Phase 1/3 を bolt-on する **拡張**。

---

## 2. 実装詳細

### 2.1 Phase 1: Plan generator

**システムプロンプト (= 新規 `phase1_prompt.py`):**

```
You are a data-analysis planner. Given a question and dataset overview,
produce a structured plan that guides a downstream agent to compute the
correct answer.

Use extended thinking to:
- Resolve ambiguous nouns by enumerating 2-3 plausible interpretations
- Identify the calculation outline (= what aggregations / joins are needed)
- Estimate the answer's structural shape (column_count, per_column,
  row_count_category, expected magnitude/range when applicable)
- Identify which files / tables / columns to focus on

Output JSON only with these keys:
{
  "column_count": <int>,
  "per_column": [<str description>, ...],
  "row_count_category": "single" | "tied-multiple" | "list-filter-subset" | "all-rows",
  "expected_row_range": "<rough estimate, e.g. '1', '5-15', '50-200'>",
  "interpretations": [
    {"term": "...", "primary": "...", "alternatives": ["...", ...], "rationale": "..."}
  ],
  "target_files": ["<rel path or table name>", ...],
  "key_filter_conditions": [
    {"column": "...", "predicate": "..."}
  ],
  "calculation_outline": "<pseudo-formula or one-sentence>",
  "expected_magnitude": "<rough scale of final value if scalar, e.g. '0-100', '1000-10000', or 'N/A'>",
  "verify_hints": [
    "<sanity check the verifier should apply, e.g. 'value should be percentage 0-100'>"
  ],

  "missing_information": [
    {
      "what": "<concrete information needed but NOT in preamble or knowledge.md>",
      "why_needed": "<which step of the calculation/filter requires it>",
      "how_to_find": "<which tool / file / heuristic should resolve it>",
      "fallback": "<what to do if the agent can't find it>"
    }
  ],

  "exploration_priorities": [
    "<step-1 priority: e.g. 'inspect_sqlite_schema on db/atom.db to confirm join key'>",
    "<step-2 priority: e.g. 'read_doc Patient.md, search for `creatinine` and extract per-patient values into a structured table'>",
    "<step-3 priority: e.g. 'execute_python: load knowledge.md, regex-search for `normal range` near WBC and FG'>"
  ]
}
```

### `missing_information` の役割

質問に答えるために必要だが **preamble + knowledge.md からは直接読み取れない**情報を Phase 1 で先に列挙させる。これにより Phase 2 の探索が "知らない知識を確実に取得しに行く" 設計になる。

具体例:

**task_344 (= medical thresholds):**
```json
"missing_information": [
  {
    "what": "Normal range for White Blood Cells (WBC) count",
    "why_needed": "Filter `WBC normal` requires explicit numeric threshold",
    "how_to_find": "execute_python: regex-scan knowledge.md and doc/Patient.md for 'WBC' near numbers. If absent, use medical literature default (4-9 ×10⁹/L)",
    "fallback": "Apply 4 ≤ WBC ≤ 9 (standard reference range). State assumption explicitly in answer thought."
  },
  {
    "what": "Abnormal threshold for Fibrinogen (FG)",
    "why_needed": "Filter `FG abnormal` requires threshold",
    "how_to_find": "Same as WBC, scan docs for FG / Fibrinogen / Fbg",
    "fallback": "Apply FG < 1.5 OR FG > 4.0 (standard reference range g/L)"
  }
]
```

**task_180 (= per unit interpretation):**
```json
"missing_information": [
  {
    "what": "What 'per unit' means in transactions context",
    "why_needed": "Filter requires 'paid more than 29.00 per unit of product 5'",
    "how_to_find": "inspect_sqlite_schema on transactions DB. Per unit likely = unit price column (Price), not total Amount",
    "fallback": "Assume 'per unit' = column 'Price' (= price per unit). State assumption."
  }
]
```

**task_169 (= date format & avg semantics):**
```json
"missing_information": [
  {
    "what": "Format of Date column in yearmonth.csv (YYYYMM vs YYYYMMDD)",
    "why_needed": "Filter year=2013 requires correct date parse",
    "how_to_find": "read_csv: yearmonth.csv first 5 rows",
    "fallback": "Try LIKE '2013%' first, observe format from samples"
  },
  {
    "what": "Whether 'monthly consumption' means avg(per-month total) or sum/12",
    "why_needed": "Calculation outline depends on interpretation",
    "how_to_find": "knowledge.md should clarify; if not, use 'AVG(yearly_total) / 12' = SUM/N×12",
    "fallback": "Use both interpretations as alternative attempts"
  }
]
```

### `exploration_priorities` の役割

Phase 2 ReAct が **最初の 3-5 step で何を必ず調べるべきか**を明示。agent が漫然と探索する代わりに、Phase 1 で識別した "missing info を埋める手順" を先に実行する。

これにより:
- task_344: Phase 2 の step 1 で knowledge.md scan、step 2 で 閾値抽出、step 3 で計算へ
- task_180: Phase 2 の step 1 で transactions schema 確認、step 2 で Price 列確認
- task_169: Phase 2 の step 1 で yearmonth.csv 内容確認 (= Date format)、step 2 で SME filter

→ **Phase 2 の探索効率が大幅向上**、step 浪費削減。

**ユーザープロンプト (= per-task):**

```
{rich_preamble}

---

Question: {task.question}

Generate the plan JSON.
```

**Phase 1 callable:**

```python
def generate_plan(task: PublicTask, preamble: str, model: ModelAdapter) -> dict:
    messages = [
        ModelMessage(role="system", content=PHASE1_SYSTEM_PROMPT),
        ModelMessage(role="user", content=f"{preamble}\n\n---\n\nQuestion: {task.question}"),
    ]
    raw = model.complete(messages, enable_thinking=True)  # 拡張 thinking
    plan = parse_plan_json(raw)  # 既存 _strip_json_fence + raw_decode 流用
    return plan
```

### 2.2 Phase 2: ReAct execution (= exp_109 ベース、plan 注入)

**変更点:**

- system prompt 末尾に `## Approved Plan (binding)` セクション追加。plan JSON 全体 (= column_count、per_column、interpretations、calculation_outline、**missing_information**、**exploration_priorities** 含む) を agent に渡す。
- agent の Rule 1 が「plan を厳守する」に変わる:

```
1. Your VERY FIRST action's `thought` must:
   a) Acknowledge the approved plan and state your interpretation choice
      (= plan.interpretations[0].primary).
   b) State which `exploration_priorities[0]` you will execute first.
   c) Confirm `column_count` and `calculation_outline` are understood.
   The plan binds your column_count, per_column, calculation_outline,
   and target_files. Deviating from the plan is a violation.
```

- 新規 Rule 20: `missing_information` の各項目を Phase 2 中に必ず resolve:

```
20. The plan's `missing_information` lists data points that are NOT
    available in the preamble — they MUST be resolved during exploration
    using the `how_to_find` strategy. If your tool calls fail to resolve
    a missing item, apply the `fallback` and state explicitly in your
    thought (e.g. "Applied fallback: assumed WBC normal = 4-9 ×10⁹/L").
    Do NOT skip missing_information items silently.
```

- 新規 Rule 21: `exploration_priorities` を順守して step 1-3 で実行:

```
21. Your FIRST 3 exploration steps SHOULD execute the plan's
    `exploration_priorities` in order. Deviation is allowed only when a
    priority is already satisfied by the preamble (state this in thought).
    This focuses exploration on the plan-identified knowledge gaps.
```

- max_steps を **32 → 24** に削減 (= plan で探索余地が絞られたぶん)

**残るルール:** 2-19 のうち、Rule 19 (INTERPRET 強制) は Phase 1 で吸収済なので削除。Rule 18 (column verify) は維持。新 20/21 を追加。

### 2.3 Phase 3: Verifier

**システムプロンプト (= 新規 `phase3_prompt.py`):**

```
You are an answer verifier. Given a plan and a candidate answer, check:

1. Column scope: does answer.column_count match plan.column_count?
   If extra columns exist, they must be dropped.

2. Row count plausibility: does the answer's row count fit
   plan.row_count_category? E.g.:
   - "single" → expect 1 row (allow 1-2 if ties)
   - "tied-multiple" → expect 1-N rows for ties
   - "list-filter-subset" → use expected_row_range as guide
   - "all-rows" → row count = source table row count (be lenient)

3. Magnitude sanity: if plan.expected_magnitude is set, does the answer
   value fit that range? E.g. percentage should be 0-100, monthly avg
   should fit data scale, etc.

4. verify_hints: apply each hint as a check.

5. **Missing information closure**: did the agent's trace resolve every
   `missing_information` item from the plan? If a fallback was applied,
   was it applied correctly? If a missing item was silently skipped,
   the answer is suspect.

Output JSON only:
{
  "verdict": "OK" | "FIX_DROP_COLUMNS" | "FIX_RECOMPUTE" | "FIX_TARGETED",
  "issues": ["<short description>", ...],
  "fix_instruction": "<one-sentence what to do, only if verdict starts with FIX>",
  "missing_info_status": [
    {"item": "<missing_info.what>", "resolved": true/false, "how": "<observed|fallback|skipped>"}
  ],
  "confidence": "high" | "medium" | "low"
}
```

**Phase 3 callable:**

```python
def verify_answer(plan: dict, answer: dict, model: ModelAdapter) -> dict:
    user_msg = (
        f"## Plan\n{json.dumps(plan, indent=2)}\n\n"
        f"## Candidate Answer\n{json.dumps(answer, indent=2)}\n\n"
        f"Verify and output JSON."
    )
    messages = [
        ModelMessage(role="system", content=PHASE3_SYSTEM_PROMPT),
        ModelMessage(role="user", content=user_msg),
    ]
    raw = model.complete(messages, enable_thinking=True)
    return parse_verdict_json(raw)
```

### 2.4 Phase 4: Repair (条件発火)

verdict ごとの分岐:

| verdict | 動作 |
|---|---|
| **OK** | answer をそのまま採用 |
| **FIX_DROP_COLUMNS** | code-level で列削除 (= LLM call 不要、`answer.columns` のうち plan に無いものを drop) |
| **FIX_RECOMPUTE** | Phase 2 ReAct loop を再起動、+4 step の budget で `fix_instruction` を初手 thought に注入 |
| **FIX_TARGETED** | Phase 2 と同じく ReAct +4 step、issue 修正限定 |

`FIX_DROP_COLUMNS` は LLM 不要 = ノーリスク fix。`FIX_RECOMPUTE` は信頼度 high のときのみ発火 (= false alarm リスクあるため)。

### 2.5 3-attempt union 維持

**注意点:** 各 attempt が独立して Phase 1+2+3+4 を実行。

```
attempt_1: Phase1 → Phase2 → Phase3 → (Phase4 if needed) → answer1
attempt_2: 同上 (T=0.6 で plan が異なる可能性)
attempt_3: 同上 (T=0.7)

merge: _signature_majority_merge(answer1, answer2, answer3)
```

= **plan ごと多様化** が attempts 間で発生 → 自然と多解釈 union (= 路線 D の片鱗) が組み込まれる。

---

## 3. 実装フェーズ

### Day 1: Skeleton + smoke

```
1. cp -r src/experiments/exp_109_plan_first_strengthen src/experiments/exp_110_plan_solve_verify
2. Add src/experiments/exp_110_plan_solve_verify/phase1_prompt.py
3. Add src/experiments/exp_110_plan_solve_verify/phase3_prompt.py
4. Modify runner.py:
   - _run_single_task_core() に Phase 1 (plan) → Phase 2 (ReAct, plan 注入) → Phase 3 (verify) → Phase 4 (repair) を追加
5. Modify prompt.py の Rule 1 (plan 厳守版に書き換え)、Rule 19 削除
6. Smoke: 3 task で動作確認
   - task_25 (= ties、Phase 1 で row_count_category="tied" を明示できるか)
   - task_169 (= calc error、Phase 3 で magnitude check 効くか)
   - task_38 (= column scope、Phase 4 FIX_DROP_COLUMNS 動作確認)
```

### Day 2: Iterate on prompts

```
- Smoke 結果から prompt 微調整
- JSON parse 失敗率を測る (target < 5%)
- Phase 1 plan 品質チェック: 50 task で Phase 1 のみ走らせて手検査
- Phase 3 verdict 分布チェック: OK 率、FIX 率、各 FIX の prevention 効果
- Single-attempt smoke 50 task 実行 (= ~30 min)
```

### Day 3: Full bench

```
- 3-attempt × N=3 bench 実行 (= ~3 hours per run × 3 = 9 hours)
- exp_109 比 per-task delta 分析
- Phase 3 の false alarm / true catch 統計
- 必要なら Phase 1/3 prompt の最終 tune
```

### Day 4 (buffer): Submit decision

```
- n=3 mean が exp_109 比 +0.02 以上なら submit (= v4)
- それ未満なら Phase 3 強化 / Phase 1 prompt 再調整 → Day 5 で再 bench
```

---

## 4. 期待値とリスク

### Best case (+0.08 vs exp_109、missing_info 機能込み)

- Phase 1 の interpretations 列挙 で task_163 の解釈 bias を解消 (+0.04)
- Phase 1 の **missing_information + fallback** で task_344 (= WBC/FG threshold) を構造的救済 (+0.02)
- Phase 1 の **exploration_priorities** で task_169 / task_180 の探索効率向上、計算/scope ミス削減 (+0.03)
- Phase 3 の magnitude check で task_169 (82M vs 460) を検出 → repair (+0.02)
- Phase 3 の column verify で task_38/259/180 の wext 削減 (+0.02)
- → **0.7328 + 0.08 = 0.81 local → LB ~0.54 = 3 位射程**

### Base case (+0.03)

- Phase 1/3 で 4-5 task 救済 → +0.04
- Phase 3 false alarm で 1-2 task regress → -0.01
- → **0.76 local → LB ~0.51 = 4 位確保**

### Worst case (-0.01)

- Phase 1 plan が agent を縛りすぎて exploration 縮小
- Phase 3 false alarm で正解を破壊
- → **0.72 local → LB ~0.48 = 5 位以下**

### リスク registers

| リスク | 確率 | 影響 | 対策 |
|---|---|---|---|
| Phase 1 JSON parse 失敗 | 中 | 中 | retry 1 回 + fallback to "no plan" mode |
| Phase 1 plan が間違い | 中 | 大 | 3-attempt で plan 多様化、interpretations で alternatives も保持 |
| Phase 3 false alarm (= 正解を FIX_RECOMPUTE で破壊) | 中 | 大 | confidence=high のみ発火、FIX_DROP_COLUMNS のみ無条件発火 |
| Phase 1+2+3 latency 増加 | 高 | 中 | Phase 1: ~8s, Phase 3: ~5s, 合計 +13s/attempt = +50% で許容範囲 |
| 3-attempt 並列で plan 全部違う方向 → union 死 | 低 | 中 | exp_109 padding fix でカバー、最悪 Phase 1 を attempt 間共有 (= shared plan, diverse exec) |

---

## 5. ファイル構成 (= 想定)

```
src/experiments/exp_110_plan_solve_verify/
├── agent.py            # exp_109 流用 + Phase 2 用 system prompt 切替
├── config.py           # exp_109 流用 (max_steps=24 に削減)
├── config.yaml         # exp_109 流用
├── preamble.py         # exp_109 (= rich preamble) 流用
├── profile.py          # exp_109 流用 (= column profile)
├── prompt.py           # Rule 1 改 (plan 厳守) + Rule 19 削除
├── phase1_prompt.py    # ★新規: PHASE1_SYSTEM_PROMPT + parse_plan_json
├── phase3_prompt.py    # ★新規: PHASE3_SYSTEM_PROMPT + parse_verdict_json
├── pipeline.py         # ★新規: run_psv_pipeline(task) → 3-phase orchestration
├── runner.py           # _run_single_task_core で pipeline.py を呼ぶ
├── runtime.py          # exp_109 流用
├── run.py              # CLI entry
└── tools/              # exp_109 と完全同一
```

新規行数推定: **~400 行** (= Phase 1/3 prompt + parser + pipeline orchestration)。

---

## 6. 計測 metrics (= 評価で確認すること)

bench 完了後の reporting で以下を集計:

1. **Score**: λ0.5 mean / std / 95% CI vs exp_109
2. **Phase 1 stats**:
   - JSON parse success rate
   - 平均 plan の column_count, interpretations 数
   - plan vs gold の column_count 一致率
   - 平均 `missing_information` items 数 / task
   - missing_information items の resolved 率 (= Phase 3 が closure 確認)
   - exploration_priorities が Phase 2 step 1-3 で実行された率
3. **Phase 3 stats**:
   - verdict 分布 (OK / FIX_DROP / FIX_RECOMPUTE / FIX_TARGETED)
   - confidence 分布
   - true catch (= verdict 正しく fix で改善) / false alarm (= 正解を壊した) 数
4. **Phase 4 stats**:
   - Phase 4 発火率
   - 発火後の score 改善 / 悪化
5. **Latency**:
   - Phase 1 / 2 / 3 / 4 の平均所要時間
   - Total per-task time vs exp_109

これらは `runner.py` の trace.json に `psv_metadata` キーで保存。

---

## 7. 着手フロー

```
[今日]
1. exp_109 → exp_110 fork
2. phase1_prompt.py / phase3_prompt.py / pipeline.py スケルトン
3. runner.py に pipeline 統合
4. smoke 3 task (= task_25 / task_169 / task_38)

[明日]
5. JSON parse 失敗率測定、prompt tune
6. 50-task single-attempt smoke
7. issues 分析、prompt 再 tune

[明後日]
8. 3-attempt × N=3 bench launch (= ~9 時間)
9. 結果分析、submit 判断
```
